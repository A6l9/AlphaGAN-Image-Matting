import torch as tch
import torch.nn as nn
import torch.nn.functional as F

from .base_loss import BaseLoss


class PerceptualLoss(BaseLoss):
    def __init__(self, features_extractor: nn.Module) -> None:
        """Initialize perceptual loss module.

        The provided `features_extractor` is expected to be a frozen feature network
        (e.g., VGG features) whose forward pass returns intermediate activations from
        multiple layers.

        Contract:
            - Input: a tensor of shape (N, C, H, W).
            - Output: a list of tensors (features), one per selected layer, ordered in
            the same order as the configured layers.

        Args:
            features_extractor: Feature extractor model (e.g., VGG-based) that returns
                multi-layer features as a list of tensors.
        """
        self.model = features_extractor

        self.criterion = nn.L1Loss()

    def __call__(self, pred: tch.Tensor, target: tch.Tensor) -> tch.Tensor:
        """Compute perceptual distance between prediction and target.

        Args:
            pred: Predicted image tensor of shape (B, C, H, W).
            target: Ground-truth image tensor of shape (B, C, H, W).

        Returns:
            float: A scalar float equal to the average L1 distance between corresponding
            feature maps extracted from `pred` and `target`.
        """
        pred_features: list[tch.Tensor] = self.model(pred)
        target_features: list[tch.Tensor] = self.model(target)

        total_loss = pred.new_zeros(()) 

        for pred_f, targ_f in zip(pred_features, target_features):
            total_loss += self.criterion(pred_f, targ_f).item()

        return total_loss / len(pred_features)
