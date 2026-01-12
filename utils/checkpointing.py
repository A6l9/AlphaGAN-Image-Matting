from datetime import datetime
from pathlib import Path

import torch as tch

from schemas import TrainComponents


def load_checkpoint(chkp_dir: Path, device: tch.device) -> dict:
    """Loads a last checkpoint 

    Args:
        chkp_dir (Path): The path to the checkpoints directory
        device (tch.device): The device to locate the checkpoint

    Returns:
        dict: The last checkpoint as a dict if it exists or empty dict otherwise
    """
    if not chkp_dir.exists():
        return {}
    
    checkpoints = sorted(
                        chkp_dir.glob("*.pth"),
                        key=lambda p: p.stat().st_mtime
                    )
    
    if not checkpoints:
        return {}
    
    last_checkpoint = tch.load(checkpoints[-1], map_location=device)

    return last_checkpoint


def make_checkpoint_dict(epoch: int, components: TrainComponents) -> dict:
    """Builds a checkpoint dictionary for saving training state:
        - epoch
        - best loss
        - generator state
        - generator optimizer state
        - generator scheduler state
        - discriminator state
        - dicriminator optimizer state
        - dicriminator scheduler state

    Args:
        epoch (int): Current epoch
        components (TrainComponents): The train components

    Returns:
        dict: The checkpoint dict
    """
    checkpoint = {
        "epoch": epoch,
        "best_loss": components.best_loss,
        "model_state": components.g_components.generator.state_dict(),
        "g_optimizer_state": components.g_components.g_optimizer.state_dict(),
        "g_scheduler_state": components.g_components.g_scheduler.state_dict(),
        "discriminator_state": components.d_components.discriminator.state_dict(),
        "d_optimizer_state": components.d_components.d_optimizer.state_dict(),
        "d_scheduler_state": components.d_components.d_scheduler.state_dict()
    }

    return checkpoint


def save_checkpoint(chkp_dir: Path, components: TrainComponents, epoch: int) -> None:
    """Saves a training checkpoint

    The checkpoint includes:
        - epoch
        - model weights
        - best test loss
        - discriminator weights
        - generator optimizer state
        - discriminator optimizer state
        - generator scheduler state
        - discriminator scheduler state

    Args:
        chkp_dir (Path): The path to the checkpoints directory
        components (TrainComponents): The train components
        epoch (int): The current epoch
    """
    if not chkp_dir.exists():
        chkp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = make_checkpoint_dict(epoch, components)

    curr_date = datetime.strftime(datetime.now(), "%d-%m-%Y_%H:%M:%S")
    checkpoint_name = f"{epoch}_{curr_date}.pth"
    checkpoint_path = chkp_dir / checkpoint_name

    tch.save(checkpoint, checkpoint_path)


def save_checkpoint_use_colab(chkp_dir: Path, 
                              components: TrainComponents, 
                              epoch: int,
                              checkpoint_name: str
                              ) -> None:
    """Saves a training checkpoint if the train executing in the GoogleColab
    This function rewrite the last and the best checkpoints for memory saving

    The checkpoint includes:
        - epoch
        - model weights
        - best test loss
        - discriminator weights
        - generator optimizer state
        - discriminator optimizer state
        - generator scheduler state
        - discriminator scheduler state

    Args:
        chkp_dir (Path): The path to the checkpoints directory
        components (TrainComponents): The train components
        epoch (int): The current epoch
        checkpoint_name (str): The name of checkpont; Usually best/last
    """
    if not chkp_dir.exists():
        chkp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = make_checkpoint_dict(epoch, components)

    checkpoint_name = f"{checkpoint_name}.pth"

    checkpoint_path = chkp_dir / checkpoint_name

    tch.save(checkpoint, checkpoint_path)
