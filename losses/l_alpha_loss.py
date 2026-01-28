import torch as tch

from .base_loss import BaseLoss


class LAlphaLoss(BaseLoss):
    def __init__(self, 
                 unknown_val: float=128.0,
                 unknown_weight: float | None=None,
                 bg_weight: float | None=None,
                 fg_weight: float | None=None,
                 weighted: bool=False) -> None:
        """Initialize alpha L1 loss with optional trimap region weighting.

        By default, this loss computes a plain per-pixel L1 distance between predicted
        and ground-truth alpha mattes.

        If `weighted=True`, the loss builds a per-pixel weight map from the trimap
        and computes a normalized weighted L1 loss. This allows you to control the relative
        importance of different trimap regions:
            - background region (trimap ~= 0.0) gets `bg_weight`
            - unknown region (trimap ~= unknown_val / 255.0) gets `unknown_weight`
            - foreground region (trimap ~= 1.0) gets `fg_weight`

        Args:
            unknown_val: Value representing the unknown (gray) region in the trimap using
                the 0/128/255 convention. It is internally converted to [0, 1] by dividing
                by 255.0.
            unknown_weight: Weight multiplier for unknown pixels when `weighted=True`.
                If None, you must provide it before calling the loss in weighted mode.
            bg_weight: Weight multiplier for known background pixels when `weighted=True`.
                If None, you must provide it before calling the loss in weighted mode.
            fg_weight: Weight multiplier for known foreground pixels when `weighted=True`.
                If None, you must provide it before calling the loss in weighted mode.
            weighted: If True, enables trimap-based region weighting. If False, the loss
                reduces to a plain mean absolute error (MAE).
        """
        super().__init__()

        self.weighted = weighted

        if self.weighted:
            self.unknown_val: float = unknown_val / 255.0
            self.unknown_weight: float = unknown_weight
            self.bg_weight: float = bg_weight
            self.fg_weight: float = fg_weight

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
    
    def get_bg_mask(self, trimap: tch.Tensor) -> tch.Tensor:
        """Return a boolean mask selecting background's pixels from the trimap.

        This mask is True where trimap equals zero and False elsewhere.

        Args:
            trimap (tch.Tensor): Trimap tensor of shape (B, 1, H, W).

        Returns:
            tch.Tensor: Boolean tensor mask of the same shape as trimap, where True indicates background's pixels.
        """
        val = tch.zeros(1, device=trimap.device, dtype=trimap.dtype)
        
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
                 pred: tch.Tensor, 
                 target: tch.Tensor, 
                 trimap: tch.Tensor | None=None
                 ) -> tch.Tensor:
        """Compute alpha L1 loss, optionally weighted by trimap regions.

        If `weighted` is disabled, this returns a plain mean absolute error (MAE)
        over all pixels.

        If `weighted` is enabled, this builds a per-pixel weight map from the trimap:
            - background pixels get weight `bg_weight`
            - unknown (gray) pixels get weight `unknown_weight`
            - foreground pixels get weight `fg_weight`

        The final loss is a normalized weighted MAE:
            sum(|pred - target| * weights) / sum(weights)

        Args:
            pred (tch.Tensor): Predicted alpha matte of shape (B, 1, H, W), values in [0, 1].
            target (tch.Tensor): Ground-truth alpha matte with the same shape as `pred`, values in [0, 1].
            trimap (tch.Tensor | None): Trimap tensor required when `weighted` is enabled.
                Expected shape (B, 1, H, W) and values in [0, 1], where:
                - background is close to 0.0
                - unknown is close to `unknown_val / 255.0`
                - foreground is close to 1.0

        Returns:
            tch.Tensor: Scalar tensor containing the (weighted) L1 alpha loss.

        Raises:
            ValueError: If `weighted` is enabled but `trimap` is None.
        """
        diff = (pred - target).abs()

        if not self.weighted:
            return diff.mean()
        
        if trimap is None:
            raise ValueError("Trimap must be provided when weighted=True.")
        
        unknown_mask = self.get_unknown_mask(trimap).to(diff.dtype)
        bg_mask = self.get_bg_mask(trimap).to(diff.dtype)
        fg_mask = self.get_fg_mask(trimap).to(diff.dtype)

        weights = tch.ones_like(diff)
        weights = weights + unknown_mask * (self.unknown_weight - 1.0)
        weights = weights + bg_mask * (self.bg_weight - 1.0)
        weights = weights + fg_mask * (self.fg_weight - 1.0)
        loss = (diff * weights).sum() / weights.sum().clamp_min(1.0)

        return loss
