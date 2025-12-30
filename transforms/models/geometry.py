import random

import torch as tch
import torchvision.transforms.v2.functional as TF
import torchvision.transforms.v2 as T

from .base import BaseTransform


class GeometryTransforms(BaseTransform):
    @staticmethod
    def resize(x: tch.Tensor, mode: T.InterpolationMode, size: int=256) -> tch.Tensor:
        """Takes x(Image as a tensor) and then resizes it by size.

        Args:
            x (tch.Tensor): Image as a tensor
            mode (T.InterpolationMode): Mode of interpolation.
            size (int, optional): Resize size. Defaults to 256.

        Returns:
            tch.Tensor: Resized image as tensors
        """
        if x.shape[0] > 3:
            x_rgb = TF.resize(x[:3], size=[size, size], interpolation=mode)
            x_trim = TF.resize(x[3:], size=[size, size], interpolation=mode)

            return tch.cat((x_rgb, x_trim), dim=0)
        return TF.resize(x, size=[size, size], interpolation=mode)
    
    @classmethod
    def random_rotate(cls,
                      orig: tch.Tensor, 
                      trim: tch.Tensor, 
                      mask: tch.Tensor,
                      angles: tuple[float, float]=(-20.0, 20.0),
                      prob: float=0.5) -> tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
        """
        Randomly rotates the foreground, trimap, and mask by an angle sampled
        from [-20°, 20°]. Uses bilinear interpolation for RGB and nearest
        for trimap/mask.

        Args:
            orig (tch.Tensor): Foreground RGB tensor of shape (3, H, W).
            trim (tch.Tensor): Trimap tensor of shape (1, H, W).
            mask (tch.Tensor): Mask tensor of shape (1, H, W).
            angles (tuple[float, float], optional): Range of rotation angles.
                Defaults to (-20.0 20.0).
            prob (float, optional): Probability of applying the rotation.
                Defaults to 0.5.

        Returns:
            tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
                Rotated (orig, trim, mask) tensors with the same dtype.
        """
        if not cls.random_apply(prob):
            return orig, trim, mask
        
        angle = random.uniform(angles[0], angles[1])

        orig_rot = TF.rotate(orig, angle, interpolation=T.InterpolationMode.BILINEAR)
        trim_rot = TF.rotate(trim, angle, interpolation=T.InterpolationMode.NEAREST)
        mask_rot = TF.rotate(mask, angle, interpolation=T.InterpolationMode.BILINEAR)

        return orig_rot, trim_rot, mask_rot
    
    @classmethod
    def random_hflip(cls,
                     orig: tch.Tensor, 
                     trim: tch.Tensor, 
                     mask: tch.Tensor,
                     prob: float=0.5) -> tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
        """
        Randomly flips the foreground, trimap, and mask horizontally.

        Args:
            orig (tch.Tensor): Foreground RGB tensor of shape (3, H, W).
            trim (tch.Tensor): Trimap tensor of shape (1, H, W).
            mask (tch.Tensor): Mask tensor of shape (1, H, W).
            prob (float, optional): Probability of applying the flip.
                Defaults to 0.5.

        Returns:
            tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
                Horizontally flipped (orig, trim, mask) tensors.
        """ 
        if not cls.random_apply(prob):
            return orig, trim, mask
        
        orig_flip = TF.hflip(orig)
        trim_flip = TF.hflip(trim)
        mask_flip = TF.hflip(mask)

        return orig_flip, trim_flip, mask_flip

    @staticmethod
    def resize_to_fit_background(
                                orig: tch.Tensor,
                                trim: tch.Tensor,
                                mask: tch.Tensor,
                                bg: tch.Tensor
                                ) -> tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
        """
        Downscales the foreground, trimap, and mask so that they fit entirely
        inside the background canvas, preserving aspect ratio.

        Args:
            bg (tch.Tensor): Background RGB tensor of shape (3, H, W).
            orig (tch.Tensor): Foreground RGB tensor of shape (3, H, W).
            trim (tch.Tensor): Trimap tensor of shape (1, H, W).
            mask (tch.Tensor): Mask tensor of shape (1, H, W).

        Returns:
            tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
                Possibly resized (orig, trim, mask) tensors.
        """
        _, h_bg, w_bg = bg.shape
        _, h_orig, w_orig = orig.shape

        scale_rate = min(1.0, w_bg / w_orig, h_bg / h_orig)
        if scale_rate < 1.0:
            # If the scale rate < 1.0, so we need to resize the orig, trimap and mask to fit it in the background 
            new_h = int(h_orig * scale_rate)
            new_w = int(w_orig * scale_rate)
            orig = TF.resize(orig, [new_h, new_w], T.InterpolationMode.BILINEAR)
            trim = TF.resize(trim, [new_h, new_w], T.InterpolationMode.NEAREST)
            mask = TF.resize(mask, [new_h, new_w], T.InterpolationMode.BILINEAR)

        return orig, trim, mask

    @classmethod
    def random_scale(cls,
                     orig: tch.Tensor, 
                     trim: tch.Tensor, 
                     mask: tch.Tensor,
                     scale: tuple[float, float]=(0.5, 0.9),
                     prob: float=0.5) -> tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
        """
        Randomly scales the foreground, trimap, and mask by a factor sampled
        from the given range.

        Args:
            orig (tch.Tensor): Foreground RGB tensor of shape (3, H, W).
            trim (tch.Tensor): Trimap tensor of shape (1, H, W).
            mask (tch.Tensor): Mask tensor of shape (1, H, W).
            scale (tuple[float, float], optional): Min and max scaling factor.
                Defaults to (0.5, 0.9).
            prob (float, optional): Probability of applying the scaling.
                Defaults to 0.5.

        Returns:
            tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
                Scaled (orig, trim, mask) tensors.
        """
        if not cls.random_apply(prob):
            return orig, trim, mask
        
        rand_scale = random.uniform(scale[0], scale[1])

        _, h, w = orig.shape

        new_h = int(h * rand_scale)
        new_w = int(w * rand_scale)

        orig_scl = TF.resize(orig, [new_h, new_w], interpolation=T.InterpolationMode.BILINEAR)
        trim_scl = TF.resize(trim, [new_h, new_w], interpolation=T.InterpolationMode.NEAREST)
        mask_scl = TF.resize(mask, [new_h, new_w], interpolation=T.InterpolationMode.BILINEAR)

        return orig_scl, trim_scl, mask_scl

    @classmethod
    def random_background_scale(cls,
                     bg: tch.Tensor,
                     len_range: tuple[int, int]=(2000, 6000)
                     ) -> tch.Tensor:
        """Randomly choose a target long-side length and compute the corresponding resize scale.

        Args:
            bg: Background image tensor in CHW format (C, H, W).
            len_range: Inclusive range (min_len, max_len) in pixels to sample the target long-side length.

        Returns:
            tch.Tensor: A resized background tensor in CHW format (C, H, W), where the resize factor
            is derived from the sampled target long-side length.
        """
        rand_len = random.randint(*len_range)

        _, h, w = bg.shape
        max_dim_idx = 1 if h >= w else 2

        scale_ratio = rand_len / bg.shape[max_dim_idx]

        new_h = int(h * scale_ratio)
        new_w = int(w * scale_ratio)

        bg_scl = TF.resize(bg, [new_h, new_w], interpolation=T.InterpolationMode.BILINEAR)

        return bg_scl
