import torch as tch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, 
                 input_chann: int, 
                 norm_layer: nn.Module, 
                 use_bias: bool=False, 
                 base_chann: int=64, 
                 n_layers: int=3) -> None:
        """Construct a PatchGAN discriminator.

        Args:
            input_chann (int): Number of channels in the input tensor
                (e.g. 4 for [RGB composite + trimap]).
            norm_layer (nn.Module): Normalization layer class to use, such as
                nn.BatchNorm2d or nn.InstanceNorm2d. It is expected to be
                callable as norm_layer(num_features).
            use_bias (bool, optional): Whether to use bias in convolutional
                layers. Defaults to False.
            base_chann (int, optional): Number of feature channels in the first
                convolutional layer. Following layers will use multiples of this
                value (e.g. 64 → 128 → 256 → 512). Defaults to 64.
            n_layers (int, optional): Number of downsampling blocks (with
                stride=2) in the discriminator. More layers increase the
                effective patch size (receptive field). Defaults to 3.
        """
        super(PatchGANDiscriminator, self).__init__()

        kernel_size = 4
        padw = 1

        sequence = [nn.Conv2d(input_chann, 
                              base_chann, 
                              kernel_size=kernel_size, 
                              stride=2, 
                              padding=padw), 
                              nn.LeakyReLU(0.2, True)]
        base_chn_mult = 1
        base_chn_mult_prev = 1
        for n_l in range(1, n_layers):
            base_chn_mult_prev = base_chn_mult
            base_chn_mult = min(2 ** n_l, 8)

            in_chann = base_chann * base_chn_mult_prev
            out_chann = base_chann * base_chn_mult

            sequence += [nn.Conv2d(in_chann, 
                                   out_chann, 
                                   kernel_size=kernel_size, 
                                   stride=2,
                                   padding=padw, 
                                   bias=use_bias), 
                                   norm_layer(out_chann), 
                                   nn.LeakyReLU(0.2, True)]

        base_chn_mult_prev = base_chn_mult
        base_chn_mult = min(2 ** n_layers, 8)
        
        in_chann = base_chann * base_chn_mult_prev
        out_chann = base_chann * base_chn_mult

        sequence += [nn.Conv2d(in_chann, 
                               out_chann, 
                               kernel_size=kernel_size, 
                               stride=1, 
                               padding=padw, 
                               bias=use_bias), 
                               norm_layer(out_chann), 
                               nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(out_chann, 1, kernel_size=kernel_size, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input: tch.Tensor) -> tch.Tensor:
        """Compute PatchGAN discriminator predictions.

        Args:
            input (tch.Tensor): Input tensor of shape (B, C, H, W), where
                C == input_chann passed to the constructor.

        Returns:
            tch.Tensor: Output tensor of shape (B, 1, H_out, W_out), containing
                raw (logits) real/fake scores for each local patch in the input.
                This output is typically fed into BCEWithLogitsLoss.
        """
        return self.model(input)
