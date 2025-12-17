import torch as tch

from .base import BaseTransform


class CropTransforms(BaseTransform):
    @classmethod
    def random_gray_crop(cls,
                         compos: tch.Tensor, 
                         trim: tch.Tensor, 
                         mask: tch.Tensor,
                         orig: tch.Tensor,
                         bg: tch.Tensor,
                         crop_size: int=256,
                         prob: float=1.0) -> tuple[tch.Tensor, 
                                                   tch.Tensor, 
                                                   tch.Tensor, 
                                                   tch.Tensor, 
                                                   tch.Tensor]:
        """
        Crops a square region of size `crop_size` centered around a random
        pixel belonging to the unknown (gray, value 128) region in the trimap.

        If the transform is not applied (by probability), the input tensors
        are returned unchanged.

        Args:
            compos (tch.Tensor): Composite RGB tensor of shape (3, H, W).
            trim (tch.Tensor): Trimap tensor of shape (1, H, W),
                with values {0, 128, 255}.
            mask (tch.Tensor): Mask tensor of shape (1, H, W).
            crop_size (int, optional): Side length of the crop. Defaults to 256.
            prob (float, optional): Probability of applying the crop.
                Defaults to 1.0.

        Raises:
            ValueError: If the trimap has no gray (128) pixels.

        Returns:
            tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
                - Cropped composite tensor of shape (3, crop_size, crop_size).
                - Cropped trimap tensor of shape (1, crop_size, crop_size).
                - Cropped mask tensor of shape (1, crop_size, crop_size).
        """
        if not cls.random_apply(prob):
            return compos, trim, mask, orig, bg
        
        _, h, w = trim.shape

        _, yt, xt = (trim == 128).nonzero(as_tuple=True)

        if len(xt) == 0:
            raise ValueError("Trimap has no gray (128) region.")

        gray_idx = tch.randint(0, len(xt), (1,))
        cx, cy = xt[gray_idx].item(), yt[gray_idx].item()

        x_left = cx - crop_size // 2
        y_left = cy - crop_size // 2

        x_left = max(0, min(x_left, w - crop_size))
        y_left = max(0, min(y_left, h - crop_size))

        crop_comp = compos[:, y_left:y_left + crop_size, x_left:x_left + crop_size]
        crop_trim = trim[:, y_left:y_left + crop_size, x_left:x_left + crop_size]
        crop_mask = mask[:, y_left:y_left + crop_size, x_left:x_left + crop_size]
        crop_orig = orig[:, y_left:y_left + crop_size, x_left:x_left + crop_size]
        crop_bg   = bg[:, y_left:y_left + crop_size, x_left:x_left + crop_size]

        return crop_comp, crop_trim, crop_mask, crop_orig, crop_bg
