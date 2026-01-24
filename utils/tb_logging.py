import typing as tp

import torch as tch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from cfg_loader import cfg


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
        _to_3ch(compos_in),
        _to_3ch(trimap_in),
        _to_3ch(alpha_orig),
        
        _to_3ch(compos_out),
        _to_3ch(trimap_out),
        _to_3ch(alpha_pred),
    ] 

    grid = make_grid(tch.stack(images, dim=0), nrow=nrow)
    writer.add_image(f"image/{tag}", grid, global_step)


def get_model_layers(model: nn.Module) -> tp.Iterator[tuple[str, nn.Module]]:
    """Yield leaf Conv2d/Conv3d/Linear layers from a model.

    This helper is intended for TensorBoard weight logging. It walks through
    `model.named_modules()` and returns only leaf modules that are instances of
    convolutional layers or linear layers.

    Args:
        model (nn.Module): Root PyTorch module to traverse.

    Yields:
        tuple[str, nn.Module]: Tuples of (layer_name, layer) for leaf Conv/Linear layers.
    """
    allowed_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)

    for layer_name, layer in model.named_modules():
        if layer_name == "":
            continue

        # Avoid returning container modules.
        if any(layer.children()):
            continue

        if isinstance(layer, allowed_types):
            yield layer_name, layer


def log_random_weights(
        model: nn.Module, 
        writer: SummaryWriter, 
        step: int,
        model_name: str,
        n_weights: int=1
        ) -> None:
    """Log random scalar weights from Conv/Linear layers to TensorBoard.

    For each Conv/Linear layer, this function samples up to `n_weights` random elements
    from the layer's weight tensor and writes them as scalars to TensorBoard.

    Args:
        model (nn.Module): Model whose weights will be logged.
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        step (int): Global step value for TensorBoard.
        model_name (str): Name used to group tags (for example 'G' or 'D').
        n_weights (int): Number of random weights to log per layer. Defaults to 1.

    Returns:
        None
    """
    from . import train_utils as trn_utl

    with trn_utl.set_seed(cfg.general.random_seed): 

        all_weights = {}
        layer_weights = []

        # Extract weights from each layer
        for layer_name, layer in get_model_layers(model):
            weights = layer.weight.detach().cpu().numpy().ravel()
            layer_weights.append((layer_name, weights))
            all_weights[layer_name] = weights

        # Choose n random weights
        for layer_name, weights in layer_weights:
            if len(weights) > 0:
                idx = np.random.choice(len(weights), size=min(n_weights, len(weights)), replace=False)
                selected_weights = weights[idx]
                for i, weight in enumerate(selected_weights):
                    tag = f"random_weights_{model_name}/{layer_name}_weight_{i}"
                    writer.add_scalar(tag, weight, global_step=step)


def compute_gradient_statistics(model: nn.Module) -> dict[str, float]:
    """Compute simple gradient magnitude statistics for a model.

    This function scans all parameters and aggregates:
        - mean_gradient: mean of abs(grad) averaged per-parameter and then averaged
          across parameters that have gradients
        - max_gradient: maximum of abs(grad) across all parameters that have gradients

    Args:
        model (nn.Module): Model whose gradients will be inspected.

    Returns:
        dict[str, float]: Dictionary with keys:
            - 'mean_gradient'
            - 'max_gradient'
        Values are floats. If no gradients are present, returns zeros.
    """
    grad_means = []
    grad_maxs = []

    for param in model.parameters():
        if param.grad is not None:
            grad_means.append(param.grad.abs().mean().item())
            grad_maxs.append(param.grad.abs().max().item())

    return {
        "mean_gradient": sum(grad_means) / len(grad_means) if grad_means else 0,
        "max_gradient": max(grad_maxs) if grad_maxs else 0
    }


def log_gradient(
        model: nn.Module, 
        writer: SummaryWriter, 
        step: int,
        model_name: str,
        ) -> None:
    """Log gradient magnitude statistics to TensorBoard.

    Computes gradient statistics via `compute_gradient_statistics` and writes them
    as scalars to TensorBoard under 'gradient_statistic_{model_name}/...'.

    Args:
        model (nn.Module): Model whose gradients will be logged.
        writer (SummaryWriter): TensorBoard SummaryWriter instance.
        step (int): Global step value for TensorBoard.
        model_name (str): Name used to group tags (for example 'G' or 'D').

    Returns:
        None
    """
    grad_stat = compute_gradient_statistics(model)

    for key, val in grad_stat.items():
        tag = f"gradient_statistic_{model_name}/{key}"        
        writer.add_scalar(tag, val, step)
