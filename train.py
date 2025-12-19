from pathlib import Path

import torch as tch
from tqdm import tqdm

import utils as utl
from cfg_loader import cfg
import schemas as sch


@tch.no_grad()
def test_one_epoch(epoch: int, loss_vals: sch.TestLossValues, train_comp: sch.TrainComponents) -> sch.TestLossValues:
    """Run one evaluation epoch.

    This function evaluates the generator on the test dataloader without gradient
    computation. It computes:
    - L1 alpha loss between predicted alpha and ground-truth alpha
    - L1 composite loss between predicted composite (built from predicted alpha) and
      the ground-truth composite RGB

    It also logs input and output images to TensorBoard.

    Args:
        epoch: Current epoch index (used for logging and global step).
        loss_vals: Accumulator for validation losses. The function adds batch losses
            to this object (as floats).
        train_comp: Training components container holding the generator, dataloaders,
            losses, device, and TensorBoard writer.

    Returns:
        sch.TestLossValues: Updated loss_vals containing accumulated losses for this validation epoch.
    """
    print(utl.color("Testing...", "green"))

    train_comp.generator.eval()

    n_batches = len(train_comp.test_loader)

    for i, batch in enumerate(train_comp.test_loader):
        step = (epoch * len(train_comp.test_loader)) + i

        compos = batch["compos"].to(train_comp.device, non_blocking=True)
        trim = batch["trim"].to(train_comp.device, non_blocking=True)
        bg = batch["bg"].to(train_comp.device, non_blocking=True)
        fg = batch["orig"].to(train_comp.device, non_blocking=True)
        mask = batch["mask"].to(train_comp.device, non_blocking=True)

        alpha_pred = train_comp.generator(compos)

        pred_compos = utl.make_compos(fg, mask, bg, alpha_pred)
        
        loss_alpha = train_comp.l_alpha_loss(pred=alpha_pred, target=mask)
        loss_comp = train_comp.l_comp_loss(
            alpha=alpha_pred, 
            fg=fg,
            bg=bg,
            target=compos[:, :3]
            )

        # Saving loss values
        loss_vals.l1_alpha_loss += float(loss_alpha.item())
        loss_vals.l1_compos_loss += float(loss_comp.item())

        # Logging input/output images every 'log_io_n_batches' batches
        if (i + 1) % cfg.train.logging.log_io_n_batches == 0:
            utl.log_matting_inputs_outputs(
                compos[:, :3],
                trim,
                mask,
                pred_compos,
                trim,
                alpha_pred,
                "test/input|output",
                step,
                train_comp.writer
            )

    # Logging loss values
    for key, value in loss_vals.__dict__.items():
        utl.log_loss(epoch, value / n_batches, f"test/{key}", train_comp.writer)
    
    return loss_vals


def train_one_epoch(epoch: int, loss_vals: sch.TrainLossValues, train_comp: sch.TrainComponents, prog_bar: tqdm) -> sch.TrainLossValues:
    """Run one training epoch for generator and discriminator.

    This function performs adversarial training in the following order per batch:
    1) Update discriminator (D) using:
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
    train_comp.generator.train()
    train_comp.discriminator.train()

    for i, batch in prog_bar:
        step = (epoch * len(train_comp.train_loader)) + i

        compos = batch["compos"].to(train_comp.device, non_blocking=True)
        trim = batch["trim"].to(train_comp.device, non_blocking=True)
        bg = batch["bg"].to(train_comp.device, non_blocking=True)
        fg = batch["orig"].to(train_comp.device, non_blocking=True)
        mask = batch["mask"].to(train_comp.device, non_blocking=True) 

        alpha_pred = train_comp.generator(compos)

        pred_compos = utl.make_compos(fg, mask, bg, alpha_pred)

        # Update D
        train_comp.d_optimizer.zero_grad(set_to_none=True)

        d_in_real = compos
        d_in_fake = utl.add_trimap(pred_compos.detach(), trim)

        d_real = train_comp.discriminator(d_in_real)
        d_fake = train_comp.discriminator(d_in_fake)

        loss_d_real = train_comp.gan_loss(pred=d_real, is_real=True)
        loss_d_fake = train_comp.gan_loss(pred=d_fake, is_real=False)

        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        loss_d.backward()
        train_comp.d_optimizer.step()
        train_comp.d_scheduler.step()

        # Update G
        train_comp.g_optimizer.zero_grad(set_to_none=True)

        d_in_fake_for_g = utl.add_trimap(pred_compos, trim)
        d_fake_for_g = train_comp.discriminator(d_in_fake_for_g)

        loss_g_gan = train_comp.gan_loss(pred=d_fake_for_g, is_real=True)
        loss_alpha = train_comp.l_alpha_loss(pred=alpha_pred, target=mask)
        loss_comp = train_comp.l_comp_loss(
            alpha=alpha_pred, 
            fg=fg,
            bg=bg,
            target=compos[:, :3]
            )
        
        # Calculate the generator loss; multiply the discriminator's loss by lambda to equalize with the rest 
        loss_g = loss_alpha + loss_comp + (loss_g_gan * cfg.train.losses.lambda_gan_g) 

        loss_g.backward()
        train_comp.g_optimizer.step()
        train_comp.g_scheduler.step()

        # Saving loss values
        loss_vals.l1_alpha_loss += float(loss_alpha.item())
        loss_vals.l1_compos_loss += float(loss_comp.item())
        loss_vals.g_loss += float(loss_g_gan.item())
        loss_vals.bce_fake_d_loss += float(loss_d_fake.item())
        loss_vals.bce_real_d_loss += float(loss_d_real.item())
        loss_vals.d_loss += float(loss_d.item())

        # Logging learning rates every 'log_lr_n_batches' batches
        if (i + 1) % cfg.train.logging.log_lr_n_batches == 0:
            utl.log_lr(
                step, 
                train_comp.d_optimizer.param_groups[0]["lr"],
                "D",
                train_comp.writer
            )
            utl.log_lr(
                step, 
                train_comp.g_optimizer.param_groups[0]["lr"],
                "G",
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
        utl.log_loss(epoch, value / len(train_comp.train_loader), f"train/{key}", train_comp.writer)
    
    return loss_vals


def train_pipeline(train_comp: sch.TrainComponents) -> None:
    """Run the full training loop: train, evaluate, and optionally save checkpoints.

    The pipeline performs epoch-based training and evaluation. It tracks the best
    validation loss and saves checkpoints according to the configured policy.

    Checkpoint policy:
    - If use_colab is False: save full checkpoints periodically and on improvement.
    - If use_colab is True: keep only two files (last and best) to save disk space.

    Args:
        train_comp: Training components including model(s), optimizers, schedulers,
            dataloaders, and training state (current epoch, best_loss).

    Side effects:
        - Updates train_comp.best_loss when validation loss improves.
        - Writes checkpoint files to cfg.general.checkpoints_dir.
    """
    train_loss_vals = sch.TrainLossValues()
    test_loss_vals = sch.TestLossValues()

    for epoch in range(train_comp.epoch, cfg.train.epoches + 1):
        # Define progress bar through the context manager
        with tqdm(iterable=enumerate(train_comp.train_loader), 
                  total=len(train_comp.train_loader), unit="batch", desc="Training...", leave=True) as prog_bar:

            prog_bar.set_description(f"Training... epoch: {epoch}")

            # Run the current epoch training
            train_loss_vals = train_one_epoch(epoch, train_loss_vals, train_comp, prog_bar)

        # Zeroing the train loss values
        train_loss_vals.zeroing_loss_values()

        # Run the current epoch testing
        test_loss_vals = test_one_epoch(epoch, test_loss_vals, train_comp)

        # Calculate 
        general_loss = test_loss_vals.l1_alpha_loss + test_loss_vals.l1_compos_loss

        # Zeroing the test loss values
        test_loss_vals.zeroing_loss_values()

        # Save checkpoint every 'save_chkp_n_epoches' batches  or if the loss was better
        if epoch % cfg.train.save_chkp_n_epoches == 0 or general_loss < train_comp.best_loss:
            print(utl.color("Saving the checkpoint...", "green"))

            chkp_dir = Path(cfg.general.checkpoints_dir).resolve()

            best = general_loss < train_comp.best_loss

            # If the loss was better save checkpoint as 'best'
            if best:
                # Update the best loss
                train_comp.best_loss = general_loss

            if not cfg.general.colab.use_colab:
                utl.save_checkpoint(chkp_dir, train_comp, epoch)
            elif cfg.general.colab.use_colab: # If using the colab the program execute another function for save checkpoint
                print(utl.color("Using a way to save checkpoints with limited memory...", "green"))

                if best:
                    utl.save_checkpoint_use_colab(chkp_dir, train_comp, epoch, cfg.general.colab.best_chkp_name)

                utl.save_checkpoint_use_colab(chkp_dir, train_comp, epoch, cfg.general.colab.last_chkp_name)
