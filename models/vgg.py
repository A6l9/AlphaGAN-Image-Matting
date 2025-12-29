import torch as tch
import torch.nn as nn
from torchvision import models


class VGG(nn.Module):
    def __init__(self, layer_indices: tuple[int, ...], model_type: str="vgg19") -> None:
        super().__init__()

        self.features = self._get_model_features(model_type)

        if not layer_indices:
            raise ValueError("layer_indices must be non-empty.")

        sorted_layer_indices = sorted(layer_indices)
        self.max_idx = sorted_layer_indices[-1]
        self.layer_indices = set(sorted_layer_indices)

        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad_(False)

    def _get_model_features(self, model_type: str) -> nn.Sequential:
        """Returns pretrained VGG features backbone.

        Args:
            model_type (str): Name of the required model e.g "vgg19"
        
        Raises:
            ValueError: Raises a ValueError error 
                if the model type not in ['vgg11', 'vgg13', 'vgg16', 'vgg19'].

        Returns:
            nn.Sequential: The VGG features backbone.
        """
        model_type = model_type.lower().strip()

        if model_type == "vgg11":
            m = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        elif model_type == "vgg13":
            m = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        elif model_type == "vgg16":
            m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif model_type == "vgg19":
            m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown model_type={model_type!r}. Use vgg11/vgg13/vgg16/vgg19.")

        return m.features

    def forward(self, x: tch.Tensor) -> list[tch.Tensor]:
        """Extract intermediate feature maps from selected layers.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            A list of feature tensors captured at the requested layer indices.
        """
        outputs = []

        for i, layer in enumerate(self.features):
            x = layer(x)

            if i in self.layer_indices:
                outputs.append(x)
            if i >= self.max_idx:
                break
        
        return outputs
