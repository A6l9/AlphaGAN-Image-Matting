import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, in_ch: int=2048, out_ch: int=256, rates: tuple=(1,12,24,36)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        ] + [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for rate in rates[1:]
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[2:]
        outs = [b(x) for b in self.branches]
        g = self.global_pool(x)
        g = F.interpolate(g, size=size, mode='bilinear', align_corners=False)
        outs.append(g)
        return self.project(torch.cat(outs, dim=1))
