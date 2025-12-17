from pathlib import Path

import torch as tch
from tqdm import tqdm

import utils as utl
from cfg_loader import cfg
import schemas as sch


def zeroing_loss_values(loss_vals: sch.LossValues) -> None:
    """
    Docstring for zeroing_loss_values
    
    :param loss_vals: Description
    :type loss_vals: sch.LossValues
    """
    loss_vals.alpha_loss = 0.0
    loss_vals.compos_loss = 0.0
    loss_vals.g_loss = 0.0
    loss_vals.fake_d_loss = 0.0
    loss_vals.real_d_loss = 0.0
    loss_vals.d_loss = 0.0


@tch.no_grad()
def test_one_epoch(epoch: int, train_comp: sch.TrainComponents) -> tuple[float, float]:
    print(utl.color("Testing...", "green"))

    train_comp.generator.eval()

    n_batches = 0
    alpha_loss_total = 0.0
    compos_loss_total = 0.0

    for i, batch in enumerate(train_comp.test_loader):
        step = (epoch * len(train_comp.train_loader)) + i

        compos = batch["compos"].to(train_comp.device, non_blocking=True)
        trim = batch["trim"].to(train_comp.device, non_blocking=True)
        bg = batch["bg"].to(train_comp.device, non_blocking=True)
        fg = batch["orig"].to(train_comp.device, non_blocking=True)
        mask = batch["mask"].to(train_comp.device, non_blocking=True)

        alpha_pred = train_comp.generator(compos)

        pred_compos = utl.make_compos(batch, alpha_pred)
        
        loss_alpha = train_comp.l_alpha_loss(pred=alpha_pred, target=mask)
        loss_comp = train_comp.l_comp_loss(
            alpha=alpha_pred, 
            fg=fg,
            bg=bg,
            target=compos[:, :3]
            )

        # Saving loss values
        alpha_loss_total += float(loss_alpha.item())
        compos_loss_total += float(loss_comp.item())

        # Logging input/output images every 10 batches
        if n_batches % 50 == 0:
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
        
        n_batches += 1

    # Logging loss values
    utl.log_loss(epoch, alpha_loss_total / n_batches, f"test/alpha_loss", train_comp.writer)
    utl.log_loss(epoch, compos_loss_total / n_batches, f"test/compos_loss", train_comp.writer)
    
    return alpha_loss_total / n_batches, compos_loss_total / n_batches


def train_one_epoch(epoch: int, loss_vals: sch.LossValues, train_comp: sch.TrainComponents, prog_bar: tqdm) -> sch.LossValues:
    """_summary_

    Args:
        epoch (int): _description_
        loss_vals (sch.LossValues): _description_
        train_comp (sch.TrainComponents): _description_

    Returns:
        sch.LossValues: _description_
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

        pred_compos = utl.make_compos(batch, alpha_pred)

        # Update D
        train_comp.d_optimizer.zero_grad(set_to_none=True)

        d_in_real = compos
        d_in_fake = utl.add_trimap(pred_compos.detach(), trim)

        d_real = train_comp.discriminator(d_in_real)
        d_fake = train_comp.discriminator(d_in_fake)

        loss_d_real = train_comp.gan_loss(pred=d_real, is_real=True)
        loss_d_fake = train_comp.gan_loss(pred=d_fake, is_real=False)

        loss_d = loss_d_real + loss_d_fake

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
        
        loss_g = loss_alpha + loss_comp + loss_g_gan

        loss_g.backward()
        train_comp.g_optimizer.step()
        train_comp.g_scheduler.step()

        # Saving loss values
        loss_vals.alpha_loss += float(loss_alpha.item())
        loss_vals.compos_loss += float(loss_comp.item())
        loss_vals.g_loss += float(loss_g_gan.item())
        loss_vals.fake_d_loss += float(loss_d_fake.item())
        loss_vals.real_d_loss += float(loss_d_real.item())
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
    
    # Zeroing the loss values
    zeroing_loss_values(loss_vals)
    
    return loss_vals


def train_pipeline(train_comp: sch.TrainComponents) -> None:
    """_summary_

    Args:
        train_comp (TrainComponents): _description_
    """
    loss_vals = sch.LossValues()

    for epoch in range(train_comp.epoch + 1, cfg.train.epoches + 1):
        # Define progress bar
        prog_bar = tqdm(iterable=enumerate(train_comp.train_loader), total=len(train_comp.train_loader), unit="batch", desc="Training...", leave=True)

        prog_bar.set_description(f"Training... epoch: {epoch}")

        train_one_epoch(epoch, loss_vals, train_comp, prog_bar)

        alpha_loss, compos_loss = test_one_epoch(epoch, train_comp)

        general_loss = alpha_loss + compos_loss
        
        # Save checkpoint every 'save_chkp_n_epoches' batches
        if epoch % cfg.train.save_chkp_n_epoches == 0 or general_loss < train_comp.best_loss:
            chkp_dir = Path(cfg.general.checkpoints_dir).resolve()

            train_comp.best_loss = general_loss

            utl.save_checkpoint(chkp_dir, train_comp, epoch)

            # Zeroing the loss values
            zeroing_loss_values(loss_vals)
