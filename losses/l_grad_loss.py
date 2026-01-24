import torch as tch

from .base_loss import BaseLoss


class LGradLoss(BaseLoss):
    def __init__(self, unknown_val: float=128.0) -> None:
        """Initialize gradient loss for alpha matte edges inside the trimap unknown region.

        Args:
            unknown_val (float): Value representing the unknown (gray) region in the trimap using
                the 0/128/255 convention. The value is internally converted to [0, 1] by
                dividing by 255.0.
        """
        super().__init__()
    
        self.unknown_val = unknown_val / 255.0

    def get_unknown_mask(self, trimap: tch.Tensor) -> tch.Tensor:
        """Return a boolean mask selecting unknown pixels from the trimap.

        This mask is True where trimap equals `self.unknown_val` and False elsewhere.

        Args:
            trimap (tch.Tensor): Trimap tensor of shape (B, 1, H, W).

        Returns:
            tch.Tensor: Boolean tensor mask of the same shape as trimap, where True indicates unknown pixels.
        """
        val = tch.tensor(self.unknown_val, device=trimap.device, dtype=trimap.dtype)
        
        return tch.isclose(trimap, val, atol=1e-4)

    def __call__(self, 
        pred: tch.Tensor, 
        target: tch.Tensor, 
        trimap: tch.Tensor
    ) -> tch.Tensor:
        """Compute gradient L1 loss between predicted and target alpha mattes in unknown region.

        Args:
            pred (tch.Tensor): Predicted alpha matte of shape (B, 1, H, W), values in [0, 1].
            target (tch.Tensor): Ground-truth alpha matte with the same shape as `pred`.
            trimap (tch.Tensor): Trimap tensor of shape (B, 1, H, W) in [0, 1], where unknown pixels
                are close to `unknown_val`.

        Returns:
            tch.Tensor: Scalar tensor containing the unknown-region gradient L1 loss.
        """
        unk_mask = self.get_unknown_mask(trimap).float()

        grad_x_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        grad_y_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        grad_x_target = target[:, :, :, 1:] - target[:, :, :, :-1]
        grad_y_target = target[:, :, 1:, :] - target[:, :, :-1, :]

        unk_mask_x = unk_mask[:, :, :, 1:]
        unk_mask_y = unk_mask[:, :, 1:, :]

        loss_x = ((grad_x_pred - grad_x_target).abs() * unk_mask_x).sum() / unk_mask_x.sum().clamp_min(1.0)
        loss_y = ((grad_y_pred - grad_y_target).abs() * unk_mask_y).sum() / unk_mask_y.sum().clamp_min(1.0)

        return loss_x + loss_y
