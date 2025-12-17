import torch as tch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .aspp import ASPP


class Encoder(nn.Module):
    def __init__(self, in_ch: int=4) -> None:
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        if in_ch != 3:
            old = resnet.conv1
            new = nn.Conv2d(
                in_ch,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=(old.bias is not None),
            )

            with tch.no_grad():
                new.weight[:, :3].copy_(old.weight)

                if in_ch > 3:
                    extra = in_ch - 3
                    new.weight[:, 3:3+extra].zero_()

                if old.bias is not None:
                    new.bias.copy_(old.bias)

            resnet.conv1 = new

        for _, m in resnet.layer3.named_modules():
            if isinstance(m, tch.nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (2, 2)
                    m.padding = (2, 2)
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
            
            if hasattr(m, "stride") and m.stride == (2, 2):
                m.stride = (1, 1)

        for _, m in resnet.layer4.named_modules():
            if isinstance(m, tch.nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.dilation = (4, 4)
                    m.padding = (4, 4)
                    if m.stride == (2, 2):
                        m.stride = (1, 1)
            if hasattr(m, "stride") and m.stride == (2, 2):
                m.stride = (1, 1)
        
        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x: tch.Tensor) -> tuple[tch.Tensor, tch.Tensor, tch.Tensor]:
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4, x0, x2 


class Decoder(nn.Module):
    def __init__(self,
                 in_ch: int=256,
                 out_ch: int=1,
                 skip_ch0: int=64,
                 skip_ch2: int=512
                 ) -> None:
        super().__init__()
        self.upsample1 = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch2, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(256 + skip_ch0, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.upsample3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x: tch.Tensor, skip0: tch.Tensor, skip2: tch.Tensor) -> tch.Tensor:
        x = tch.cat([x, skip2], dim=1)
        x = self.upsample1(x)

        x = tch.cat([x, skip0], dim=1)
        x = self.upsample2(x)

        x = self.upsample3(x)
        
        x = self.final(x)

        return x 
    

class AlphaGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.aspp = ASPP()
        self.decoder = Decoder()
    
    def forward(self, x: tch.Tensor) -> tch.Tensor:
        x, skip0, skip2 = self.encoder(x)
        x = self.aspp(x)
        x = self.decoder(x, skip0, skip2)

        return x
