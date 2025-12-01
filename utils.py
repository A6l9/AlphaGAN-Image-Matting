import random
import typing as tp
import multiprocessing as mp
from contextlib import contextmanager

import torch as tch


@contextmanager
def set_seed(seed: int) -> tp.Generator:
    """Temporarily sets the random seed for Python's `random`,
    NumPy (if used), PyTorch CPU RNG, and CUDA RNG.

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
