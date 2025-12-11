import random
from datetime import datetime
import typing as tp
import multiprocessing as mp
from pathlib import Path
from contextlib import contextmanager

import torch as tch

from schemas import TrainComponents


COLORS = {
    "red":     "\033[31m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "blue":    "\033[34m",
    "white":   "\033[37m",
    "reset":   "\033[0m",
}


@contextmanager
def set_seed(seed: int) -> tp.Generator:
    """Temporarily sets the random seed for Python's `random`, 
    PyTorch CPU RNG, and CUDA RNG.

    This context manager ensures fully deterministic behavior
    inside the `with` block while preserving and restoring all
    previous RNG states afterwards.

    Args:
        seed (int): The random seed to use temporarily.

    Example:
        with temporary_seed(42):
            # deterministic random operations here
            x = torch.rand(1)
    # Outside the block, RNG states are restored.
    """
    random_state = random.getstate()
    torch_state = tch.get_rng_state()
    cuda_states = None
    if tch.cuda.is_available():
        cuda_states = tch.cuda.get_rng_state_all()

    try:
        random.seed(seed)
        tch.manual_seed(seed)
        if tch.cuda.is_available():
            tch.cuda.manual_seed_all(seed)

        tch.backends.cudnn.deterministic = True
        tch.backends.cudnn.benchmark = False

        yield

    finally:
        random.setstate(random_state)
        tch.set_rng_state(torch_state)
        if tch.cuda.is_available():
            tch.cuda.set_rng_state_all(cuda_states)

        tch.backends.cudnn.deterministic = False
        tch.backends.cudnn.benchmark = True


def get_num_workers() -> int:
    """Calculates the number of workers.

    Returns:
        int: The number of workers
    """
    num_cpu = mp.cpu_count()
    num_workers = min(2, num_cpu)

    return num_workers


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


def save_checkpoint(chkp_dir: Path, components: TrainComponents, epoch: int) -> None:
    """Saves a training checkpoint

    The checkpoint includes:
        - epoch
        - model weights
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

    checkpoint = {
        "epoch": epoch,
        "model_state": components["model"].state_dict(),
        "discriminator_state": components["discriminator"].state_dict(),
        "g_optimizer_state": components["g_optimizer"].state_dict(),
        "d_optimizer_state": components["d_optimizer"].state_dict(),
        "g_scheduler_state": components["g_scheduler"].state_dict(),
        "d_scheduler_state": components["d_scheduler"].state_dict()
    }

    curr_date = datetime.strftime(datetime.now(), "%d-%m-%Y_%H:%M:%S")
    checkpoint_name = f"{epoch}_{curr_date}.pth"
    checkpoint_path = chkp_dir / checkpoint_name

    tch.save(checkpoint, checkpoint_path)


def color(text: str, name: str) -> str:
    """Colorizes the terminal output

    Args:
        text (str): Terminal output
        name (str): The name of the color

    Returns:
        str: The colorized teriminal output
    """
    return f"{COLORS.get(name, COLORS['reset'])}{text}{COLORS['reset']}"
