import random

import numpy as np
import torch as tch
from PIL import Image
from torchvision.transforms.v2 import functional as F


class CustomTransforms:
    @staticmethod
    def random_apply(prob: float=0.5) -> bool:
        """
        Returns True with probability `prob`.

        Args:
            prob (float, optional): Probability of returning True. Defaults to 0.5.

        Returns:
            bool: Whether the transformation should be applied.
        """
        return random.random() < prob

    @classmethod
    def random_gray_crop(cls,
                         compos: tch.Tensor, 
                         trim: tch.Tensor, 
                         mask: tch.Tensor, 
                         crop_size: int=256,
                         prob: float=0.5) -> tuple[tch.Tensor, tch.Tensor]:
        """
        Crops a random `crop_size x crop_size` region from the composite and mask tensors.
        The crop is centered around a randomly chosen pixel belonging to the gray (128)
        unknown region in the trimap. If the trimap contains no gray pixels, an error is raised.

        Args:
            compos (tch.Tensor): Composite tensor of shape (C, H, W).
            trim (tch.Tensor): Trimap tensor of shape (H, W) or (1, H, W).
            mask (tch.Tensor): Mask tensor of shape (C, H, W).
            crop_size (int, optional): Side length of the square crop. Defaults to 256.
            prob (float, optional): Probability to apply this transform. Defaults to 0.5.

        Raises:
            ValueError: If the trimap contains no gray (128) region.

        Returns:
            tuple[tch.Tensor, tch.Tensor]:
                - Cropped composite tensor of shape (C, crop_size, crop_size).
                - Cropped mask tensor of shape (C, crop_size, crop_size).
        """
        if not cls.random_apply(prob):
            return compos, mask
        
        h, w = trim.shape

        yt, xt = (trim == 128).nonzero(as_tuple=True)

        if len(xt) == 0:
            raise ValueError("Trimap has no gray (128) region.")

        gray_idx = tch.randint(0, len(xt), (1,))
        cx, cy = xt[gray_idx].item(), yt[gray_idx].item()

        x_left = cx - crop_size // 2
        y_left = cy - crop_size // 2

        x_left = max(0, min(x_left, w - crop_size))
        y_left = max(0, min(y_left, h - crop_size))

        crop_comp = compos[:, y_left:y_left + crop_size, x_left:x_left + crop_size]
        crop_mask = mask[:, y_left:y_left + crop_size, x_left:x_left + crop_size]

        return crop_comp, crop_mask

    @classmethod
    def random_rotate(cls,
                      orig: Image.Image, 
                      trim: Image.Image, 
                      mask: Image.Image,
                      prob: float=0.5) -> tuple[Image.Image, Image.Image, Image.Image]:
        """
        Applies a random rotation in the range [-20°, 20°] to the input PIL images.
        Rotation is applied identically to `orig`, `trim`, and `mask`.

        Note:
            This transformation must be applied before composing the foreground
            onto the background, because composition requires PIL images.

        Args:
            orig (Image.Image): Foreground PIL image.
            trim (Image.Image): Trimap PIL image.
            mask (Image.Image): Mask PIL image.
            prob (float, optional): Probability to apply rotation. Defaults to 0.5.

        Returns:
            tuple[Image.Image, Image.Image, Image.Image]:
                Rotated versions of (orig, trim, mask).
        """
        if not cls.random_apply(prob):
            return orig, trim, mask
        
        angle = random.uniform(-20, 20)

        orig_rot = orig.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
        trim_rot = trim.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
        mask_rot = mask.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)

        return orig_rot, trim_rot, mask_rot
    
    @classmethod
    def random_scale(cls,
                     orig: Image.Image, 
                     trim: Image.Image, 
                     mask: Image.Image,
                     scale: tuple[float, float]=(0.5, 0.9),
                     prob: float=0.5) -> tuple[Image.Image, Image.Image, Image.Image]:
        """
        Applies a random scaling to the foreground, trimap, and mask PIL images.
        The scaling factor is uniformly sampled from the given range and applied
        identically to all three images.

        Note:
            This transformation must be applied before composing the foreground
            onto the background, because composition requires PIL images.

        Args:
            orig (Image.Image): The foreground PIL image.
            trim (Image.Image): The trimap PIL image.
            mask (Image.Image): The binary mask PIL image.
            scale (tuple[float, float], optional): Range from which the scaling factor
                is sampled. Defaults to `(0.5, 0.9)`.
            prob (float, optional): Probability of applying the transform.
                Defaults to `0.5`.

        Returns:
            tuple[Image.Image, Image.Image, Image.Image]:
                The scaled `(orig, trim, mask)` images.
        """
        if not cls.random_apply(prob):
            return orig, trim, mask
        
        rand_scale = random.uniform(scale[0], scale[1])

        w, h = orig.size

        new_h = int(h * rand_scale)
        new_w = int(w * rand_scale)

        orig_s = orig.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        trim_s = trim.resize((new_w, new_h), resample=Image.Resampling.NEAREST)
        mask_s = mask.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

        return orig_s, trim_s, mask_s
    
    @classmethod
    def random_hflip(cls,
                     orig: Image.Image, 
                     trim: Image.Image, 
                     mask: Image.Image,
                     prob: float=0.5) -> tuple[Image.Image, Image.Image, Image.Image]:
        """
        Applies a random horizontal flip to the foreground, trimap, and mask PIL images.
        All three images are flipped consistently to preserve spatial alignment.

        Note:
            This transformation must be applied before composing the foreground
            onto the background, because composition requires PIL images.

        Args:
            orig (Image.Image): The foreground PIL image.
            trim (Image.Image): The trimap PIL image.
            mask (Image.Image): The binary or soft mask PIL image.
            prob (float, optional): Probability of applying the flip.
                Defaults to `0.5`.

        Returns:
            tuple[Image.Image, Image.Image, Image.Image]:
                The horizontally flipped `(orig, trim, mask)` images.
        """
        if not cls.random_apply(prob):
            return orig, trim, mask
        
        orig = orig.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        trim = trim.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        return orig, trim, mask
