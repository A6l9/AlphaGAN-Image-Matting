import torch as tch
from PIL import Image
import torchvision.transforms.functional as TF

from .base import BaseTransform
from cfg_loader import cfg


class NormalizeTransforms(BaseTransform):
    @staticmethod
    def to_tensor(x: Image.Image) -> tch.Tensor:
        """
        Converts a PIL image to a torch tensor in (C, H, W) layout.

        Args:
            x (Image.Image): Input PIL image.

        Returns:
            tch.Tensor: Image tensor of shape (C, H, W) with dtype uint8.
        """
        return TF.pil_to_tensor(x)
    
    @classmethod
    def normalize(cls,
                x: tch.Tensor
                ) -> tch.Tensor:
        """Scale an image-like tensor to [0, 1].

        Args:
            x (tch.Tensor): Input tensor, typically in CHW format. Common cases:
                - RGB image: shape (3, H, W), values in [0, 255]
                - Composite with trimap: shape (4, H, W), values in [0, 255]
                The dtype can be uint8 or float; the output is always float32.

        Returns:
            tch.Tensor: Normalized tensor with the same shape as 'x':
                - Always scaled to [0, 1]
        """
        x = x.float() / 255.0

        return x

    @classmethod
    def imgnet_normalize(
        cls,
        x: tch.Tensor,
        mean: list[float],
        std: list[float]
    ) -> tch.Tensor:
        """Apply ImageNet normalization to RGB channels.

        Supports inputs with 3 channels (RGB) or 4 channels where the last channel is kept
        unchanged (for example RGB+extra).

        Args:
            x: Input tensor of shape (B, C, H, W), where C is 3 or 4.
            mean: Per-channel ImageNet mean for RGB.
            std: Per-channel ImageNet std for RGB.

        Returns:
            Normalized tensor with the same shape as x.

        Raises:
            ValueError: If x does not have 4 dimensions or C is not 3 or 4.
        """
        if x.ndim != 4:
            raise ValueError("Expected x with shape (B, C, H, W).")
        
        mean_ten = x.new_tensor(mean).view(1, -1, 1, 1)
        std_ten = x.new_tensor(std).view(1, -1, 1, 1)

        _, x_chn, _, _ = x.shape

        if x_chn == 4:
            x_rgb = x[:, :3]
            x_extra = x[:, 3:]

            x_rgb_norm = (x_rgb - mean_ten) / std_ten

            return tch.cat([x_rgb_norm, x_extra], dim=1)
        elif x_chn == 3:
            return (x - mean_ten) / std_ten
        
        raise ValueError("Expected x with 3 or 4 channels.")
