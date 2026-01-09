import torch as tch
from tqdm import tqdm

import utils as utl
from cfg_loader import cfg
import schemas as sch


def update_generator(
                    fg: tch.Tensor,
                    bg: tch.Tensor,
                    mask: tch.Tensor,
                    compos: tch.Tensor,
                    pred_compos: tch.Tensor, 
                    trim: tch.Tensor, 
                    alpha_pred: tch.Tensor, 
                    train_comp: sch.TrainComponents
                    ) -> sch.GLosses:
    """Run one optimization step for the generator.

    Computes the supervised matting losses:
    - alpha L1 loss between predicted alpha and ground-truth mask
    - composite L1 loss between predicted composite (using alpha_pred) and target RGB

    If a discriminator is enabled, also adds the GAN loss that encourages the
    discriminator to classify generated composites as real. The final generator
    loss is a weighted sum of these terms.

    Args:
        fg (tch.Tensor): Foreground RGB tensor of shape (B, 3, H, W).
        bg (tch.Tensor): Background RGB tensor of shape (B, 3, H, W).
        mask (tch.Tensor): Ground-truth alpha/mask tensor of shape (B, 1, H, W).
        compos (tch.Tensor): Input composite tensor (may include trimap channel),
            expected shape (B, C, H, W) where C is 3 or 4.
        pred_compos (tch.Tensor): Predicted composite RGB tensor of shape (B, 3, H, W).
        trim (tch.Tensor): Trimap tensor of shape (B, 1, H, W).
        alpha_pred (tch.Tensor): Predicted alpha tensor of shape (B, 1, H, W).
        train_comp (sch.TrainComponents): Training state containing the generator,
            losses, optimizer, scheduler, and optional discriminator components.

    Returns:
        sch.GLosses: Scalar loss values for logging (alpha, composite, and optional GAN).
    """
    train_comp.g_components.g_optimizer.zero_grad(set_to_none=True)

    with train_comp.amp_components.autocast:
        loss_alpha = train_comp.l_alpha_loss(pred=alpha_pred, target=mask)
        loss_comp = train_comp.l_comp_loss(
            alpha=alpha_pred, 
            fg=fg,
            bg=bg,
            target=compos[:, :3]
            )
    
    loss_g_gan = 0.0
    
    if train_comp.use_gan_loss:
        d_in_fake_for_g = utl.add_trimap(pred_compos, trim)

        with train_comp.amp_components.autocast:
            d_fake_for_g = train_comp.d_components.discriminator(d_in_fake_for_g)

            loss_g_gan = train_comp.d_components.gan_loss(pred=d_fake_for_g, is_real=True)
    
        # Calculate the weighted generator loss;
        weighted_alpha = (loss_alpha * cfg.train.losses.lambda_alpha_g)
        weighted_comp = (loss_comp * cfg.train.losses.lambda_comp_g)
        weighted_gan = (loss_g_gan * cfg.train.losses.lambda_gan_g)

        loss_g = weighted_alpha + weighted_comp + weighted_gan
    else:
        # Calculate the weighted generator loss without the gan loss
        weighted_alpha = (loss_alpha * cfg.train.losses.lambda_alpha_g)
        weighted_comp = (loss_comp * cfg.train.losses.lambda_comp_g)

        loss_g = weighted_alpha + weighted_comp

    # Do the backward step of the optimizer through the grad scaler if it is enabled
    if train_comp.amp_components.grad_scaler is None:
        loss_g.backward()
        train_comp.g_components.g_optimizer.step()
    else:
        train_comp.amp_components.grad_scaler.scale(loss_g).backward()
        train_comp.amp_components.grad_scaler.step(train_comp.g_components.g_optimizer)
    train_comp.g_components.g_scheduler.step()

    g_losses = sch.GLosses(
                alpha_loss=float(loss_alpha.item()),
                compos_loss=float(loss_comp.item())
            )
    
    if train_comp.use_gan_loss:
        g_losses.gan_loss = float(loss_g_gan.item())
    
    return g_losses


def update_discriminator(
                         compos: tch.Tensor,
                         pred_compos: tch.Tensor,
                         trim: tch.Tensor,
                         train_comp: sch.TrainComponents
                         ) -> sch.DLosses:
    """Run one optimization step for the discriminator.

    Args:
        compos (tch.Tensor): Real composite tensor fed to the discriminator, shape (B, C, H, W).
        pred_compos (tch.Tensor): Generated composite RGB tensor of shape (B, 3, H, W).
        trim (tch.Tensor): Trimap tensor of shape (B, 1, H, W).
        train_comp (sch.TrainComponents): Training state containing discriminator components,
            GAN loss, optimizer, and scheduler.

    Returns:
        sch.DLosses: Scalar loss values for logging (real loss, fake loss, and total D loss).
    """
    train_comp.d_components.d_optimizer.zero_grad(set_to_none=True)

    d_in_real = compos
    d_in_fake = utl.add_trimap(pred_compos.detach(), trim)

    with train_comp.amp_components.autocast:
        d_real = train_comp.d_components.discriminator(d_in_real)
        d_fake = train_comp.d_components.discriminator(d_in_fake)

        loss_d_real = train_comp.d_components.gan_loss(pred=d_real, is_real=True)
        loss_d_fake = train_comp.d_components.gan_loss(pred=d_fake, is_real=False)

    loss_d = 0.5 * (loss_d_real + loss_d_fake)

    if train_comp.amp_components.grad_scaler is None:
        loss_d.backward()
        train_comp.d_components.d_optimizer.step()
    else:
        train_comp.amp_components.grad_scaler.scale(loss_d).backward()
        train_comp.amp_components.grad_scaler.step(train_comp.d_components.d_optimizer)
    train_comp.d_components.d_scheduler.step()

    return sch.DLosses(
        loss_d_real=float(loss_d_real.item()),
        loss_d_fake=float(loss_d_fake.item()),
        loss_d=float(loss_d.item())
    )


def train_one_epoch(epoch: int, loss_vals: sch.TrainLossValues, train_comp: sch.TrainComponents, prog_bar: tqdm) -> sch.TrainLossValues:
    """Run one training epoch for generator and discriminator.

    This function performs adversarial training in the following order per batch:
    1) Update discriminator (D) (if train_comp.d_components != None) using:
       - real composite+trimap as positive samples
       - generated composite+trimap (detached) as negative samples
    2) Update generator (G) using a weighted sum of:
       - alpha L1 loss (alpha_pred vs alpha_gt)
       - composite L1 loss (composite(alpha_pred) vs composite_gt RGB)
       - GAN loss (make D classify generated composites as real)

    It also logs scalars (losses, learning rates) and images to TensorBoard
    according to the configured batch intervals.

    Args:
        epoch: Current epoch index (used for global step).
        loss_vals: Accumulator for training losses. Batch losses are added as floats.
        train_comp: Training components container with models, optimizers, schedulers,
            losses, device, and writer.
        prog_bar: Progress bar iterator over the training dataloader.

    Returns:
        sch.TrainLossValues: Updated loss_vals containing accumulated losses for the epoch.
    """
    train_comp.g_components.generator.train()

    if train_comp.use_gan_loss:
        train_comp.d_components.discriminator.train()

    for i, batch in prog_bar:
        step = (epoch * len(train_comp.train_loader)) + i

        compos = batch["compos"].to(train_comp.device, non_blocking=True)
        trim = batch["trim"].to(train_comp.device, non_blocking=True)
        bg = batch["bg"].to(train_comp.device, non_blocking=True)
        fg = batch["orig"].to(train_comp.device, non_blocking=True)
        mask = batch["mask"].to(train_comp.device, non_blocking=True) 

        with train_comp.amp_components.autocast:
            alpha_pred = train_comp.g_components.generator(compos)

        pred_compos = utl.make_compos(fg, mask, bg, alpha_pred)

        # Update D if train_comp.d_components != None and if a batch index % cfg.train.D.update_n_batches == 0
        if train_comp.use_gan_loss and (i + 1) % cfg.train.D.update_n_batches == 0:
            d_losses = update_discriminator(compos, pred_compos, trim, train_comp)

        # Update G
        g_losses = update_generator(fg, bg, mask, compos, pred_compos, trim, alpha_pred, train_comp)

        # Update grad scaler if it enabled
        if train_comp.amp_components.grad_scaler is not None:
            train_comp.amp_components.grad_scaler.update()

        # Saving loss values
        loss_vals.l1_alpha_loss += g_losses.alpha_loss
        loss_vals.l1_compos_loss += g_losses.compos_loss

        # Saving GAN and D losses if train_comp.d_components != None and if a batch index % cfg.train.D.update_n_batches == 0
        if train_comp.use_gan_loss and (i + 1) % cfg.train.D.update_n_batches == 0:
            loss_vals.g_loss += g_losses.gan_loss
            loss_vals.bce_fake_d_loss += d_losses.loss_d_fake
            loss_vals.bce_real_d_loss += d_losses.loss_d_real
            loss_vals.d_loss += d_losses.loss_d

        # Logging current G losses every 'log_curr_losses_n_batches'
        if (i + 1) % cfg.train.logging.log_curr_loss_n_batches == 0:
            for key, value in g_losses.__dict__.items():
                utl.log_loss(step, value, f"curr_loss_train/G/{key}", train_comp.writer)
        
        # Check is D enabled or not
        if train_comp.use_gan_loss and (i + 1) % cfg.train.D.update_n_batches == 0:
            # Logging current D losses every 'log_curr_losses_n_batches'
            if (i + 1) % cfg.train.logging.log_curr_loss_n_batches == 0:
                for key, value in d_losses.__dict__.items():
                    utl.log_loss(step, value, f"curr_loss_train/D/{key}", train_comp.writer)

        # Logging learning rates every 'log_lr_n_batches' batches
        if (i + 1) % cfg.train.logging.log_lr_n_batches == 0:
            utl.log_lr(
                step, 
                train_comp.g_components.g_optimizer.param_groups[0]["lr"],
                "G",
                train_comp.writer
            )

            if train_comp.use_gan_loss and (i + 1) % cfg.train.D.update_n_batches == 0:
                utl.log_lr(
                step, 
                train_comp.d_components.d_optimizer.param_groups[0]["lr"],
                "D",
                train_comp.writer
            )
        
        # Logging input/output images every 'save_io_n_batches' batches
        if (i + 1) % cfg.train.logging.log_io_n_batches == 0:
            utl.log_matting_inputs_outputs(
                compos[:, :3],
                trim,
                mask,
                pred_compos,
                trim,
                alpha_pred,
                "train/input|output",
                step,
                train_comp.writer
            )
    
    # Logging loss values
    for key, value in loss_vals.__dict__.items():
        utl.log_loss(epoch, value / len(train_comp.train_loader), f"loss_train/{key}", train_comp.writer)
    
    return loss_vals
