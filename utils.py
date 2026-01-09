import random
from datetime import datetime
import typing as tp
import multiprocessing as mp
from pathlib import Path
import contextlib as ctxlb

import torch as tch
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from schemas import TrainComponents
from cfg_loader import cfg


COLORS = {
    "red":     "\033[31m",
    "green":   "\033[32m",
    "yellow":  "\033[33m",
    "blue":    "\033[34m",
    "white":   "\033[37m",
    "reset":   "\033[0m",
}


@ctxlb.contextmanager
def set_seed(seed: int, use_cuda: bool=True) -> tp.Generator:
    """Temporarily sets RNG seeds for deterministic behavior.

    This context manager saves and restores RNG states while temporarily setting the
    random seed for Python's `random` and PyTorch CPU RNG. Optionally, it can also
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
    torch_state = tch.get_rng_state()
    cuda_states = None

    use_cuda = use_cuda and tch.cuda.is_available()

    if use_cuda:
        cuda_states = tch.cuda.get_rng_state_all()
    
    cudnn_det = tch.backends.cudnn.deterministic
    cudnn_bench = tch.backends.cudnn.benchmark

    try:
        random.seed(seed)
        tch.manual_seed(seed)
        if use_cuda:
            tch.cuda.manual_seed_all(seed)

        tch.backends.cudnn.deterministic = True
        tch.backends.cudnn.benchmark = False

        yield

    finally:
        random.setstate(random_state)
        tch.set_rng_state(torch_state)
        if use_cuda and cuda_states is not None:
            tch.cuda.set_rng_state_all(cuda_states)

        tch.backends.cudnn.deterministic = cudnn_det
        tch.backends.cudnn.benchmark = cudnn_bench


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
        "model_state": components.generator.state_dict(),
        "g_optimizer_state": components.g_optimizer.state_dict(),
        "g_scheduler_state": components.g_scheduler.state_dict(),
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


def color(text: str, name: str) -> str:
    """Colorizes the terminal output

    Args:
        text (str): Terminal output
        name (str): The name of the color

    Returns:
        str: The colorized teriminal output
    """
    return f"{COLORS.get(name, COLORS['reset'])}{text}{COLORS['reset']}"


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


@tch.no_grad()
def denorm_imagenet_rgb(x: tch.Tensor) -> tch.Tensor:
    """Undo ImageNet normalization for an RGB tensor.

    This function applies the inverse of ImageNet-style normalization:

        y = x * std + mean

    where 'mean' and 'std' are taken from 'cfg.general.mean' and
    'cfg.general.std'.

    Args:
        x (tch.Tensor): ImageNet-normalized RGB tensor in CHW format
            (3, H, W). The tensor is expected to be on any device; mean/std
            will be moved to 'x.device'.

    Returns:
        tch.Tensor: De-normalized RGB tensor in CHW format (3, H, W), float32.
            Values are typically in the [0, 1] range if the original image was
            scaled to [0, 1] before normalization.
    """
    mean = tch.tensor(cfg.general.mean).view(-1,1,1).to(x.device)
    std = tch.tensor(cfg.general.std).view(-1,1,1).to(x.device)

    return x * std + mean


def log_loss(
            epoch: int,
            loss_value: float, 
            tag: str, 
            writer: SummaryWriter
            ) -> None:
    """Logs the loss value for an epoch 

    Args:
        epoch (int): The current epoch
        loss_value (float): The loss value
        tag (str): The tag for showing in the tb
        writer (SummaryWriter): TensorBoard SummaryWriter instance used for logging.
    """
    writer.add_scalar(tag, loss_value, epoch)


def log_lr(
           global_step: int, 
           lr_value: float, 
           lr_owner: str, 
           writer: SummaryWriter
           ) -> None:
    """Log a learning rate value to TensorBoard.

    Args:
        global_step (int): Global step index used as the x-axis in TensorBoard.
            Typically computed as epoch * num_batches
        lr_value (float): Learning rate value to log.
        lr_owner (str): Identifier of what the LR belongs to, e.g. "D" for
            discriminator, "G" for generator.
        writer (SummaryWriter): TensorBoard SummaryWriter instance used for logging.
    """
    writer.add_scalar(f"lr_rate/{lr_owner}", lr_value, global_step)


@tch.no_grad()
def log_matting_inputs_outputs(
    compos_in: tch.Tensor,
    trimap_in: tch.Tensor,
    alpha_orig: tch.Tensor,
    compos_out: tch.Tensor,
    trimap_out: tch.Tensor,
    alpha_pred: tch.Tensor,
    tag: str, 
    global_step: int,
    writer: SummaryWriter,
    nrow: int=3
    ) -> None:
    """Log a 2x3 grid of matting inputs and outputs to TensorBoard.

    This utility visualizes a single sample (first item in batch if input is BCHW)
    as a panel composed of six images:

        Row 1: composite input, trimap input, ground-truth alpha
        Row 2: composite output, trimap output, predicted alpha

    The function converts tensors to float CPU tensors in CHW format.

    Args:
        compos_in (tch.Tensor): Input composite image. Shape CHW or BCHW.
        trimap_in (tch.Tensor): Input trimap. Shape HW, 1HW, CHW, or batched variants.
        alpha_orig (tch.Tensor): Ground-truth alpha matte. Shape HW, 1HW, CHW, or batched variants.
        compos_out (tch.Tensor): Сomposite image with using the predicted alpha. Shape CHW or BCHW.
        trimap_out (tch.Tensor): Trimap associated with the output (often the same as input). Shape compatible with trimap_in.
        alpha_pred (tch.Tensor): Predicted alpha matte. Shape HW, 1HW, CHW, or batched variants.
        tag (str): TensorBoard tag suffix. Final tag will be 'image/{tag}'.
        global_step (int): Global step used as the x-axis value in TensorBoard.
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        nrow (int): Number of images per row in the grid. Use 3 to get a 2x3 layout.
    """
    def _to_chw(x: tch.Tensor) -> tch.Tensor:
        if x.dim() == 4:
            x = x[0]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.detach().float().cpu()

        return x

    def _to_3ch(x: tch.Tensor) -> tch.Tensor:
        x = _to_chw(x)

        if x.size(0) == 4:
            x = x[:3]

        if x.size(0) == 1:
            x = x.repeat(3, 1, 1)

        return x
    
    images = [
        denorm_imagenet_rgb(_to_3ch(compos_in)),
        _to_3ch(trimap_in),
        _to_3ch(alpha_orig),
        denorm_imagenet_rgb(_to_3ch(compos_out)),
        _to_3ch(trimap_out),
        _to_3ch(alpha_pred),
    ] 

    grid = make_grid(tch.stack(images, dim=0), nrow=nrow)
    writer.add_image(f"image/{tag}", grid, global_step)



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
