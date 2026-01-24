import torch as tch
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
