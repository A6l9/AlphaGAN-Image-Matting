import torch as tch

from .base import BaseTransform


class CropTransforms(BaseTransform):
    @classmethod
    def random_unknown_crop(cls,
                         compos: tch.Tensor, 
                         trim: tch.Tensor, 
                         mask: tch.Tensor,
                         orig: tch.Tensor,
                         bg: tch.Tensor,
                         crop_size: int=256,
                         unknown_val: int=128,
                         prob: float=1.0
                         ) -> tuple[tch.Tensor, 
                                    tch.Tensor, 
                                    tch.Tensor, 
                                    tch.Tensor, 
                                    tch.Tensor]:
        """Crop a square patch centered around a random pixel from the trimap unknown region.

        Args:
            compos (tch.Tensor): Composite RGB tensor of shape (3, H, W).
            trim (tch.Tensor): Trimap tensor of shape (1, H, W). Common values are
                {0, unknown_value, 255}.
            mask (tch.Tensor): Ground-truth alpha/mask tensor of shape (1, H, W).
            orig (tch.Tensor): Foreground RGB tensor aligned with 'compos', shape (3, H, W).
            bg (tch.Tensor): Background RGB tensor aligned with 'compos', shape (3, H, W).
            crop_size (int, optional): Side length of the square crop in pixels.
                Defaults to 256.
            prob (float, optional): Probability of applying this transform.
                Defaults to 1.0.
            unknown_value (int, optional): Trimap value that denotes the unknown region
                used for sampling the crop center. Defaults to 128.

        Raises:
            ValueError: If the trimap contains no pixels with value 'unknown_value'.

        Returns:
            tuple[tch.Tensor, tch.Tensor, tch.Tensor, tch.Tensor, tch.Tensor]:
                - crop_comp (tch.Tensor): Cropped composite tensor, shape (3, crop_size, crop_size).
                - crop_trim (tch.Tensor): Cropped trimap tensor, shape (1, crop_size, crop_size).
                - crop_mask (tch.Tensor): Cropped mask tensor, shape (1, crop_size, crop_size).
                - crop_orig (tch.Tensor): Cropped foreground tensor, shape (3, crop_size, crop_size).
                - crop_bg (tch.Tensor): Cropped background tensor, shape (3, crop_size, crop_size).
        """ 
        if not cls.random_apply(prob):
            return compos, trim, mask, orig, bg
        
        _, h, w = trim.shape

        _, yt, xt = (trim == unknown_val).nonzero(as_tuple=True)

        if len(xt) == 0:
            raise ValueError(f"Trimap has no unknown ({unknown_val}) region.")

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
