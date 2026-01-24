import random
import typing as tp
import multiprocessing as mp
import contextlib as ctxlb

import torch as tch
from torch import nn
import numpy as np
from torch.amp.grad_scaler import GradScaler


@ctxlb.contextmanager
def set_seed(seed: int, use_cuda: bool=True) -> tp.Generator:
    """Temporarily sets RNG seeds for deterministic behavior.

    This context manager saves and restores RNG states while temporarily setting the
    random seed for Python's `random`, numpy's `random` and PyTorch CPU RNG. Optionally, it can also
    set and restore CUDA RNG states.

    Args:
        seed: The random seed to use temporarily.
        use_cuda: If True, also seed and restore CUDA RNG states.

    Example:
        with set_seed(42, use_cuda=False):
            # Deterministic CPU-side random operations here
            ...
        # Outside the block, RNG states are restored.
    """
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    torch_state = tch.get_rng_state()
    cuda_states = None

    use_cuda = use_cuda and tch.cuda.is_available()

    if use_cuda:
        cuda_states = tch.cuda.get_rng_state_all()
    
    cudnn_det = tch.backends.cudnn.deterministic
    cudnn_bench = tch.backends.cudnn.benchmark

    try:
        np.random.seed(seed)
        random.seed(seed)
        tch.manual_seed(seed)
        if use_cuda:
            tch.cuda.manual_seed_all(seed)

        tch.backends.cudnn.deterministic = True
        tch.backends.cudnn.benchmark = False

        yield

    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        tch.set_rng_state(torch_state)
        if use_cuda and cuda_states is not None:
            tch.cuda.set_rng_state_all(cuda_states)

        tch.backends.cudnn.deterministic = cudnn_det
        tch.backends.cudnn.benchmark = cudnn_bench


def get_num_workers(reserve_cpus: int = 1, max_workers: int | None = None) -> int:
    """Calculates a reasonable number of worker processes.

    Args:
        reserve_cpus: How many CPU cores to keep free.
        max_workers: Optional upper bound for workers.

    Returns:
        The number of workers to use (at least 1).
    """
    cpu = mp.cpu_count()
    workers = max(1, cpu - reserve_cpus)
    if max_workers is not None:
        workers = min(workers, max_workers)
    return workers


def make_compos(fg: tch.Tensor, mask: tch.Tensor, bg: tch.Tensor, alpha: tch.Tensor) -> tch.Tensor:
    """Compose a foreground over a background using a predicted alpha matte.

    The foreground is first gated by the ground-truth mask (to suppress any
    pixels outside the object), then alpha-blended onto the background using
    the predicted alpha:

        comp = alpha * (fg * mask) + (1 - alpha) * bg

    Args:
        fg (tch.Tensor): Foreground RGB tensor of shape (B, 3, H, W) or (3, H, W).
        mask (tch.Tensor): Ground-truth mask/alpha tensor used to gate the foreground.
            Shape (B, 1, H, W) or (1, H, W). Expected to be in [0, 1].
        bg (tch.Tensor): Background RGB tensor of shape (B, 3, H, W) or (3, H, W).
        alpha (tch.Tensor): Predicted alpha matte used for blending.
            Shape (B, 1, H, W) or (1, H, W). Expected to be in [0, 1].

    Returns:
        tch.Tensor: Composite RGB tensor with the same spatial size as `bg`,
            shape (B, 3, H, W) or (3, H, W).
    """
    return alpha * (fg * mask) + (1.0 - alpha) * bg


def add_trimap(compos: tch.Tensor, trimap: tch.Tensor) -> tch.Tensor:
    """Adds the trimap as a 4th channel to the compos

    Args:
        compos (tch.Tensor): The composite of the foreground and the background images
        trimap (tch.Tensor): The trimap of the foreground image

    Returns:
        tch.Tensor: The composite with the trimap
    """

    return tch.cat([compos, trimap], dim=1)


def get_amp_dtype_from_str(dtype: str) -> tch.dtype:
    """Convert a config string into a torch dtype used by AMP autocast.

    Supported values:
    - "bf16" -> torch.bfloat16
    - "fp16" -> torch.float16

    Args:
        dtype: AMP dtype string from config.

    Returns:
        tch.dtype: The corresponding `tch.dtype` to pass into `tch.autocast`.

    Raises:
        ValueError: If `dtype` is not one of the supported values.
    """
    key = dtype.strip().lower()

    if key == "bf16":
        return tch.bfloat16
    if key == "fp16":
        return tch.float16
    
    raise ValueError(f"Unsupported AMP dtype: {dtype}. Use 'bf16' or 'fp16'.")


def make_autocast(enabled: bool, device: tch.device, dtype: str) -> tch.autocast | ctxlb.nullcontext:
    """Create an autocast context manager, or a no-op context when AMP is disabled.

    Args:
        enabled: Whether AMP is enabled in the config.
        device: Target device used for training (must be CUDA for autocast to apply).
        dtype: AMP dtype string ("bf16" or "fp16").

    Returns:
        tch.autocast | ctxlb.nullcontext: `nullcontext()` when AMP is disabled or the device is not CUDA
                                          `torch.autocast(...)` otherwise
    """
    if not enabled or device.type != "cuda":
        return ctxlb.nullcontext()
    
    return tch.autocast(device_type="cuda", dtype=get_amp_dtype_from_str(dtype))


def make_grad_scaler(
    enabled: bool,
    device: tch.device,
    dtype: str,
    use_grad_scaler: bool,
    ) -> GradScaler | None:
    """Create a GradScaler when using fp16 AMP, otherwise return None.

    Args:
        enabled: Whether AMP is enabled in the config.
        device: Target device used for training (must be CUDA for GradScaler).
        dtype: AMP dtype string ("bf16" or "fp16").
        use_grad_scaler: Whether to use GradScaler (usually True for fp16, False for bf16).

    Returns:
        GradScaler | None: `GradScaler` instance if fp16 scaling is enabled, otherwise None.
    """
    if not enabled or device.type != "cuda":
        return None
    if not use_grad_scaler:
        return None
    if dtype.strip().lower() != "fp16":
        return None
    
    return GradScaler()


def set_requires_grad(model: nn.Module, status: bool) -> None:
    """Enable or disable gradient computation for all parameters of a module.

    Args:
        model (nn.Module): PyTorch module whose parameters will be updated.
        status (bool): If True, enables gradients (requires_grad=True) for all parameters.
            If False, disables gradients (requires_grad=False) for all parameters.
    """
    for par in model.parameters():
        par.requires_grad_(status)
