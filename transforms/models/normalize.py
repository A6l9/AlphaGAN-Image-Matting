import torch as tch
from PIL import Image
import torchvision.transforms.functional as TF

from .base import BaseTransform
from cfg_loader import cfg


class NormalizeTransforms(BaseTransform):
    mean = tch.tensor(cfg.general.mean).view(-1,1,1)
    std = tch.tensor(cfg.general.std).view(-1,1,1)

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
                x: tch.Tensor,
                imgnet: bool=False
                ) -> tch.Tensor:
        """Scale an image-like tensor to [0, 1] and optionally apply ImageNet normalization.

        The function always converts the input to 'float32' and scales values from
        '[0, 255]' to '[0, 1]' via division by 255.

        If 'imgnet=True':
        - For 3-channel tensors (RGB), ImageNet mean/std normalization is applied
            to all channels.
        - For 4+ channel tensors where the first 3 channels are RGB (e.g. RGB + trimap),
            ImageNet normalization is applied only to the first 3 channels, while the
            remaining channels are kept in the '[0, 1]' range.

        Args:
            x (tch.Tensor): Input tensor, typically in CHW format. Common cases:
                - RGB image: shape (3, H, W), values in [0, 255]
                - Composite with trimap: shape (4, H, W), values in [0, 255]
                The dtype can be uint8 or float; the output is always float32.
            imgnet (bool): If True, apply ImageNet mean/std normalization to RGB channels.
                Defaults to False.

        Returns:
            tch.Tensor: Normalized tensor with the same shape as 'x':
                - Always scaled to [0, 1]
                - Additionally ImageNet-normalized on RGB channels when 'imgnet=True'
        """
        # Firstly perform x to float type and range [0, 1]
        x = x.float() / 255.0

        if imgnet:
            # Perform x to ImageNet format(only to RGB part)
            if x.shape[0] == 3:
                return (x - cls.mean) / cls.std

            trim = x[3:]

            x_imgnet = (x[:3] - cls.mean) / cls.std

            return tch.cat((x_imgnet, trim), dim=0)
        
        return x
