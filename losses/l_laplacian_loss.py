import torch as tch
import torch.nn.functional as F

from .base_loss import BaseLoss


class LAlphaLaplacianLoss(BaseLoss):
    def __init__(self, device: tch.device, dtype: tch.dtype=tch.float32) -> None:
        """Initialize Laplacian loss for alpha matte detail supervision.

        Args:
            device (tch.device): Device where the Laplacian kernel will live.
            dtype (tch.dtype, optional): Kernel dtype. Defaults to tch.float32
        """
        self.kernel = tch.tensor(
                                [[0.0,  1.0, 0.0],
                                [1.0, -4.0, 1.0],
                                [0.0,  1.0, 0.0]],
                                device=device,
                                dtype=dtype
                                ).view(1, 1, 3, 3)
        
    def unknown_mask_from_trimap(self, trimap: tch.Tensor, unknown_val: int=128) -> tch.Tensor:
        """Build a boolean mask for the unknown (gray) region from a trimap.

        Args:
            trimap: Trimap tensor of shape (B, 1, H, W).
            unknown_val: Unknown value in [0,255]. Defaults to 128.

        Returns:
            tch.Tensor: Boolean tensor mask with the same spatial shape as trimap, where True marks
            unknown pixels.
        """

        val = unknown_val / 255.0

        return tch.isclose(trimap, tch.tensor(val, device=trimap.device), atol=1e-4)

    def apply_laplacian_2d(self, x: tch.Tensor) -> tch.Tensor:
        """Apply a 2D Laplacian filter to a batch of alpha mattes.

        Args:
            x: Input tensor of shape (B, 1, H, W).

        Returns:
            tch.Tensor: Tensor of shape (B, 1, H, W) containing Laplacian responses.
        """
        return F.conv2d(x, self.kernel, padding=1)

    def __call__(
        self,
        pred: tch.Tensor,
        target: tch.Tensor,
        trimap: tch.Tensor
    ) -> tch.Tensor:
        """Compute Laplacian L1 loss on the unknown region of a trimap.

        Steps:
        1) Filter pred and target alpha mattes with Laplacian.
        2) Compute per-pixel absolute difference between Laplacian responses.
        3) Keep only unknown pixels using the trimap mask.
        4) Normalize by the number of unknown pixels.

        Args:
            pred: Predicted alpha tensor of shape (B, 1, H, W).
            target: Ground-truth alpha tensor of shape (B, 1, H, W).
            trimap: Trimap tensor used to define unknown region. Shape (B, 1, H, W). Expected encoding is normalized 0..1.

        Returns:
            tch.Tensor: Scalar tensor loss.
        """
        lap_pred = self.apply_laplacian_2d(pred)
        lap_tar = self.apply_laplacian_2d(target)
        diff = (lap_pred - lap_tar).abs()

        unknown_mask = self.unknown_mask_from_trimap(trimap).float()
        amount_true = unknown_mask.sum().clamp_min(1.0)

        return (diff * unknown_mask).sum() / amount_true
