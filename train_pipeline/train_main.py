from pathlib import Path

from tqdm import tqdm

import utils as utl
from cfg_loader import cfg
import schemas as sch
from .test_one_epoch import test_one_epoch
from .train_one_epoch import train_one_epoch


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

        # Calculate general test loss
        general_loss = test_loss_vals.percept_loss

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
