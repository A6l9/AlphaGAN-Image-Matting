import random

import torch as tch

from .base import BaseTransform


class CompositingTransforms(BaseTransform):
    @staticmethod
    def random_placement(
                         orig: tch.Tensor,
                         trim: tch.Tensor,
                         mask: tch.Tensor,
                         bg: tch.Tensor
                         ) -> tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
        """
        Randomly places the foreground onto the background.

        Args:
            orig (tch.Tensor): Foreground RGB tensor of shape (3, H, W).
            trim (tch.Tensor): Trimap tensor of shape (1, H, W).
            mask (tch.Tensor): Alpha/mask tensor of shape (1, H, W) with values in [0, 255].
            bg (tch.Tensor): Background RGB tensor of shape (3, H, W).

        Returns:
            tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
                - Composite RGB tensor of shape (3, H, W).
                - Full-size trimap tensor of shape (1, H, W).
                - Full-size mask tensor of shape (1, H, W).
        """
        _, h_bg, w_bg = bg.shape

        obj_mask = mask > 0
        _, y_nz, x_nz = obj_mask.nonzero(as_tuple=True) # Takes indexes all the nonzero pixels

        if y_nz.numel() == 0:
            raise ValueError("The mask should not be empty")
        
        ymin, ymax = y_nz.min().item(), y_nz.max().item()
        xmin, xmax = x_nz.min().item(), x_nz.max().item()

        orig_obj = orig[:, ymin:ymax+1, xmin:xmax+1]
        trim_obj = trim[:, ymin:ymax+1, xmin:xmax+1]
        mask_obj = mask[:, ymin:ymax+1, xmin:xmax+1]

        _, h_orig, w_orig = orig_obj.shape

        x_rand = random.randint(0, w_bg - w_orig)
        y_rand = random.randint(0, h_bg - h_orig)

        alpha = mask_obj.float() / 255.0

        orig_area = bg[:, y_rand:y_rand+h_orig, x_rand:x_rand+w_orig]
        
        comp = orig_obj * alpha + orig_area * (1 - alpha)

        final = bg.clone()
        final[:, y_rand:y_rand+h_orig, x_rand:x_rand+w_orig] = comp

        trim_big = tch.zeros((1, h_bg, w_bg))
        mask_big = tch.zeros((1, h_bg, w_bg))
        trim_big[:, y_rand:y_rand+h_orig, x_rand:x_rand+w_orig] = trim_obj
        mask_big[:, y_rand:y_rand+h_orig, x_rand:x_rand+w_orig] = mask_obj

        return final, trim_big, mask_big

    @staticmethod
    def concat_image_and_trimap(comp: tch.Tensor, trim: tch.Tensor) -> tch.Tensor:
        """
        Concatenates RGB composite and trimap along the channel dimension.

        Args:
            comp (tch.Tensor): Composite RGB tensor of shape (3, H, W).
            trim (tch.Tensor): Trimap tensor of shape (1, H, W).

        Returns:
            tch.Tensor: Tensor of shape (4, H, W), where the last channel is trimap.
        """
        concatenated = tch.cat((comp, trim), dim=0)

        return concatenated
