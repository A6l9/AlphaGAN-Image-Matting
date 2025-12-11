import torch as tch
import torch.nn.functional as F

from .base_loss import BaseLoss


class LCompositeLoss(BaseLoss):
    def composite(self, 
                  alpha: tch.Tensor,
                  fg: tch.Tensor,
                  bg: tch.Tensor
                  ) -> tch.Tensor:
        """Compose a predicted image from foreground, background and alpha matte.

        Args:
            alpha (tch.Tensor): Predicted alpha matte of shape (B, 1, H, W) in the range [0, 1].
            fg (tch.Tensor): Foreground image tensor of shape (B, C, H, W).
            bg (tch.Tensor): Background image tensor of shape (B, C, H, W).

        Returns:
            tch.Tensor: Composited image tensor of shape (B, C, H, W).
        """
        comp_pred = alpha * fg + (1 - alpha) * bg

        return comp_pred

    def __call__(self, 
                    alpha: tch.Tensor,
                    fg: tch.Tensor,
                    bg: tch.Tensor, 
                    target: tch.Tensor) -> tch.Tensor:
        """
        Compute the compositional L1 loss between predicted and target composites.

        This loss first reconstructs a composite image using the predicted alpha
        matte and the provided foreground/background (via `composite`), and then
        measures the L1 distance to the target composite image.

        Args:
            alpha (tch.Tensor): Predicted alpha matte, same shape rules as in
                `composite`.
            fg (tch.Tensor): Foreground image tensor of shape (B, C, H, W).
            bg (tch.Tensor): Background image tensor of shape (B, C, H, W).
            target (tch.Tensor): Ground-truth composite image tensor of shape (B, C, H, W)

        Returns:
            tch.Tensor: Scalar tensor containing the L1 compositional loss value.
        """
        comp_pred = self.composite(alpha, fg, bg)

        loss = F.l1_loss(comp_pred, target)

        return loss
