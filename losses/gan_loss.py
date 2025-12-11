import torch as tch
import torch.nn as nn

from .base_loss import BaseLoss


class GANLoss(BaseLoss):
    """
    Adversarial loss for PatchGAN-style discriminator using BCEWithLogitsLoss.

    This class can be used both:
      - for the discriminator (is_real=True / False)
      - for the generator (is_real=True for fake samples).
    """

    def __init__(self, real_label: float = 1.0, fake_label: float = 0.0) -> None:
        """Initialize GAN loss.

        Args:
            real_label (float): Target value for 'real' samples. Defaults to 1.0.
            fake_label (float): Target value for 'fake' samples. Defaults to 0.0.
        """
        self.real_label = real_label
        self.fake_label = fake_label
        self.criterion = nn.BCEWithLogitsLoss()

    def __getitem__(self, pred: tch.Tensor, is_real: bool) -> tch.Tensor:
        """Compute adversarial loss for given predictions and target type.

        Args:
            pred (tch.Tensor): Discriminator output logits of shape (B, 1, H, W).
            is_real (bool): If True, use 'real' targets (label=real_label),
                otherwise use 'fake' targets (label=fake_label).

        Returns:
            tch.Tensor: Scalar tensor with BCE adversarial loss.
        """
        if is_real:
            target_val = self.real_label
        else:
            target_val = self.fake_label

        target = tch.full_like(pred, fill_value=target_val)
        loss = self.criterion(pred, target)

        return loss
