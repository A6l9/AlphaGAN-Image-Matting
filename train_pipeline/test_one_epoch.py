import torch as tch

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

    train_comp.g_components.generator.eval()

    n_batches = len(train_comp.test_loader)

    for i, batch in enumerate(train_comp.test_loader):
        step = (epoch * len(train_comp.test_loader)) + i

        compos = batch["compos"].to(train_comp.device, non_blocking=True)
        trim = batch["trim"].to(train_comp.device, non_blocking=True)
        bg = batch["bg"].to(train_comp.device, non_blocking=True)
        fg = batch["orig"].to(train_comp.device, non_blocking=True)
        mask = batch["mask"].to(train_comp.device, non_blocking=True)

        with train_comp.amp_components.autocast:
            alpha_pred = train_comp.g_components.generator(compos)

            pred_compos = utl.make_compos(fg, mask, bg, alpha_pred)
            
            loss_alpha = train_comp.l_alpha_loss(pred=alpha_pred, target=mask)
            loss_comp = train_comp.l_comp_loss(
                alpha=alpha_pred, 
                fg=fg,
                bg=bg,
                target=compos[:, :3]
                )
            percept_loss = train_comp.percept_loss(pred=pred_compos[:, :3], target=compos[:, :3])
        
        # Logging current metrics every 'log_curr_mets_n_batches'
        if (i + 1) % cfg.test.logging.log_curr_mets_n_batches == 0:
            utl.log_loss(step, float(loss_alpha.item()), f"curr_mets_test/alpha_loss", train_comp.writer)
            utl.log_loss(step, float(loss_comp.item()), f"curr_mets_test/compos_loss", train_comp.writer)
            utl.log_loss(step, float(percept_loss.item()), f"curr_mets_test/percept_loss", train_comp.writer)

        # Saving loss values
        loss_vals.l1_alpha_loss += float(loss_alpha.item())
        loss_vals.l1_compos_loss += float(loss_comp.item())
        loss_vals.percept_loss += float(percept_loss.item())

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
        utl.log_loss(epoch, value / n_batches, f"loss_test/{key}", train_comp.writer)
    
    return loss_vals
