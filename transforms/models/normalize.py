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
                compos: tch.Tensor,
                mask: tch.Tensor) -> tuple[tch.Tensor, tch.Tensor]:
        """
        Normalizes the composite RGB + trimap image and scales the mask to [0, 1].

        - Composite is converted to float in [0, 1] and normalized
        using ImageNet-style mean/std.
        - Mask is converted to float in [0, 1] (assuming original [0, 255]).

        Args:
            compos (tch.Tensor): Composite RGB + trimap tensor of shape (4, H, W),
                values in [0, 255].
            mask (tch.Tensor): Mask tensor of shape (1, H, W),
                values in [0, 255].

        Returns:
            Returns:
                tuple[tch.Tensor, tch.Tensor]:
                    - Normalized composite tensor of shape (4, H, W), float32
                        (3 normalized RGB channels + 1 trimap channel in [0, 1]).
                    - Mask tensor of shape (1, H, W) in [0, 1], float32.
        """
        # Firstly perform composite and mask to float type and range [0, 1]
        compos = compos.float() / 255.0 
        mask = mask.float() / 255.0

        trim = compos[3:]
        # Then perform composite to ImageNet format(only to RGB part)
        compos_norm = (compos[:3] - cls.mean) / cls.std

        return tch.cat((compos_norm, trim), dim=0), mask
