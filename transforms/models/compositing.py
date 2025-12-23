import random

import torch as tch

from .base import BaseTransform


class CompositingTransforms(BaseTransform):
    @staticmethod
    def random_placement(
                         orig: tch.Tensor,
                         trim: tch.Tensor,
                         mask: tch.Tensor,
                         bg: tch.Tensor,
                         device: tch.device
                         ) -> tuple[tch.Tensor, 
                                    tch.Tensor, 
                                    tch.Tensor, 
                                    tch.Tensor, 
                                    tch.Tensor]:
        """
        Randomly places the foreground onto the background.

        Args:
            orig (tch.Tensor): Foreground RGB tensor of shape (3, H, W).
            trim (tch.Tensor): Trimap tensor of shape (1, H, W).
            mask (tch.Tensor): Alpha/mask tensor of shape (1, H, W) with values in [0, 255].
            bg (tch.Tensor): Background RGB tensor of shape (3, H, W).
            device (tch.device): Device on which tensors will be allocated

        Returns:
            tuple[tch.Tensor, tch.Tensor, tch.Tensor, tch.Tensor, tch.Tensor]:
                - Composite RGB tensor of shape (3, H, W).
                - Full-size trimap tensor of shape (1, H, W).
                - Full-size mask tensor of shape (1, H, W).
                - Full-size foreground tensor of shape (3, H, W)
                - Background RGB tensor of shape (3, H, W).
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

        trim_big = tch.zeros((1, h_bg, w_bg), device=device)
        mask_big = tch.zeros((1, h_bg, w_bg), device=device)
        trim_big[:, y_rand:y_rand+h_orig, x_rand:x_rand+w_orig] = trim_obj
        mask_big[:, y_rand:y_rand+h_orig, x_rand:x_rand+w_orig] = mask_obj

        orig_big = tch.zeros_like(bg, device=device)
        orig_big[:, y_rand:y_rand+h_orig, x_rand:x_rand+w_orig] = orig_obj

        return final, trim_big, mask_big, orig_big, bg

    @staticmethod
    def center_placement(
                         orig: tch.Tensor,
                         trim: tch.Tensor,
                         mask: tch.Tensor,
                         bg: tch.Tensor,
                         device: tch.device
                         ) -> tuple[tch.Tensor, 
                                    tch.Tensor, 
                                    tch.Tensor, 
                                    tch.Tensor, 
                                    tch.Tensor]:
        """Place the foreground object at the center of the background and build a composite sample.

        Args:
            orig (tch.Tensor): Foreground RGB image tensor of shape (3, H, W).
            trim (tch.Tensor): Foreground trimap tensor of shape (1, H, W).
            mask (tch.Tensor): Foreground mask/alpha tensor of shape (1, H, W).
                Expected to be in [0, 255] (uint8-like) or already scaled accordingly.
            bg (tch.Tensor): Background RGB image tensor of shape (3, H_bg, W_bg).
            device (tch.device): Target device for newly created tensors.

        Raises:
            ValueError: If the mask has no non-zero pixels (empty object region).

        Returns:
            tuple[tch.Tensor, tch.Tensor, tch.Tensor, tch.Tensor, tch.Tensor]:
                A 5-tuple containing:
                - final (tch.Tensor): Composite RGB image of shape (3, H_bg, W_bg).
                - trim_big (tch.Tensor): Trimap placed into background canvas, shape (1, H_bg, W_bg).
                - mask_big (tch.Tensor): Mask placed into background canvas, shape (1, H_bg, W_bg).
                - orig_big (tch.Tensor): Foreground RGB placed into background canvas, shape (3, H_bg, W_bg).
                - bg (tch.Tensor): The original background tensor (same object as input `bg`).
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

        x_cent = (w_bg - w_orig) // 2
        y_cent = (h_bg - h_orig) // 2

        alpha = mask_obj.float() / 255.0

        orig_area = bg[:, y_cent:y_cent+h_orig, x_cent:x_cent+w_orig]
        
        comp = orig_obj * alpha + orig_area * (1 - alpha)

        final = bg.clone()
        final[:, y_cent:y_cent+h_orig, x_cent:x_cent+w_orig] = comp

        trim_big = tch.zeros((1, h_bg, w_bg), device=device)
        mask_big = tch.zeros((1, h_bg, w_bg), device=device)
        trim_big[:, y_cent:y_cent+h_orig, x_cent:x_cent+w_orig] = trim_obj
        mask_big[:, y_cent:y_cent+h_orig, x_cent:x_cent+w_orig] = mask_obj

        orig_big = tch.zeros_like(bg, device=device)
        orig_big[:, y_cent:y_cent+h_orig, x_cent:x_cent+w_orig] = orig_obj

        return final, trim_big, mask_big, orig_big, bg

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
