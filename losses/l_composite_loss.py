import torch as tch
import torch.nn.functional as F

from .base_loss import BaseLoss


class LCompositeLoss(BaseLoss):
    def __init__(self, 
                 unknown_val: float=128.0,
                 unknown_weight: float | None=None,
                 fg_weight: float | None=None,
                 weighted: bool=False) -> None:
        """Initialize compositional L1 loss, optionally weighting trimap regions.

        By default, this loss computes a plain mean absolute error (MAE) between:
            comp_pred = alpha * fg + (1 - alpha) * bg
        and the target composite RGB.

        If `weighted=True`, the loss uses the trimap to build a per-pixel weight map:
            - unknown (gray) pixels get weight `unknown_weight`
            - known foreground pixels get weight `fg_weight`
            - all other pixels get weight 1.0

        Args:
            unknown_val (float): Value representing the unknown (gray) region in the trimap using
                the 0/128/255 convention. It is internally converted to [0, 1] by dividing
                by 255.0.
            unknown_weight (float): Weight multiplier for unknown pixels when `weighted=True`.
                If None, you must provide it before calling the loss in weighted mode.
            fg_weight (float): Weight multiplier for known foreground pixels when `weighted=True`.
                If None, you must provide it before calling the loss in weighted mode.
            weighted (bool): If True, enables trimap-based region weighting. If False, computes
                a plain MAE over all pixels.
        """
        super().__init__()

        self.weighted = weighted

        if self.weighted:
            self.unknown_val: float = unknown_val / 255.0
            self.unknown_weight: float = unknown_weight
            self.fg_weight: float = fg_weight

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
    
    def get_fg_mask(self, trimap: tch.Tensor) -> tch.Tensor:
        """Return a boolean mask selecting fg's pixels from the trimap.

        This mask is True where trimap equals 1.0 and False elsewhere.

        Args:
            trimap (tch.Tensor): Trimap tensor of shape (B, 1, H, W).

        Returns:
            tch.Tensor: Boolean tensor mask of the same shape as trimap, where True indicates fg's pixels.
        """
        val = tch.ones(1, device=trimap.device, dtype=trimap.dtype)
        
        return tch.isclose(trimap, val, atol=1e-4)

    def __call__(self, 
                    alpha: tch.Tensor,
                    fg: tch.Tensor,
                    bg: tch.Tensor,
                    target: tch.Tensor,
                    trimap: tch.Tensor | None = None) -> tch.Tensor:
        """Compute compositional L1 loss between predicted and target composites.

        First reconstructs the predicted composite using the provided `fg`, `bg`,
        and the predicted `alpha`:
            comp_pred = alpha * fg + (1 - alpha) * bg

        Then computes an L1 difference to `target`. If `weighted=False`, returns
        a plain mean absolute error over all pixels.

        If `weighted=True`, builds a per-pixel weight map from the trimap and computes
        a normalized weighted L1 loss:
            loss = sum(|comp_pred - target| * weights) / sum(weights)

        Weight map definition when `weighted=True`:
            - unknown pixels (trimap ~= unknown_val / 255.0) have weight `unknown_weight`
            - known foreground pixels (trimap ~= 1.0) have weight `fg_weight`
            - all other pixels have weight 1.0

        Args:
            alpha (tch.Tensor): Predicted alpha matte of shape (B, 1, H, W), values in [0, 1].
            fg (tch.Tensor): Foreground RGB tensor of shape (B, 3, H, W).
            bg (tch.Tensor): Background RGB tensor of shape (B, 3, H, W).
            target (tch.Tensor): Ground-truth composite RGB tensor of shape (B, 3, H, W).
            trimap (tch.Tensor | None): Trimap tensor of shape (B, 1, H, W), values in [0, 1]. Required when `weighted=True`.
                Defaults: None
        Returns:
            tch.Tensor: Scalar tensor containing the (weighted) compositional L1 loss.

        Raises:
            ValueError: If `weighted=True` but `trimap` is None.
        """
        comp_pred = self.composite(alpha, fg, bg)

        diff = (comp_pred - target).abs()

        if not self.weighted:
            return diff.mean()
        
        if trimap is None:
            raise ValueError("Trimap must be provided when weighted=True.")
        
        unknown_mask = self.get_unknown_mask(trimap).to(diff.dtype)
        fg_mask = self.get_fg_mask(trimap).to(diff.dtype)

        weights = tch.ones_like(diff)
        weights = weights + unknown_mask * (self.unknown_weight - 1.0)
        weights = weights + fg_mask * (self.fg_weight - 1.0)
        loss = (diff * weights).sum() / weights.sum().clamp_min(1.0)

        return loss
