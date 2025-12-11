import torch as tch
import torch.nn.functional as F

from .base_loss import BaseLoss


class LAlphaLoss(BaseLoss):
    def __call__(self, pred: tch.Tensor, target: tch.Tensor) -> tch.Tensor:
        """Compute alpha loss as an L1 distance between predicted and ground-truth alpha mattes.

        This loss encourages the predicted alpha matte to be close to the ground-truth alpha
        in a per-pixel sense.

        Args:
            pred (tch.Tensor): Predicted alpha matte of shape (B, C, H, W) in the range [0, 1].
            target (tch.Tensor): Ground-truth alpha matte with the same shape as `pred`.

        Returns:
            tch.Tensor: Scalar tensor containing the L1 loss value between `pred` and `target`.
        """
        loss = F.l1_loss(pred, target)

        return loss
