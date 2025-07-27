import torch
from torch import nn
from torch.nn import functional as F

class Upsample(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        # (B,C,H,W) ->  (B,C,H*2,W*2)
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        return self.conv(x)