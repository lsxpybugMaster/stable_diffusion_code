# VAE Decoder
import torch
from torch import nn
from torch.nn import functional as F
from vae_blocks import VAE_AttentionBlock,VAE_ResidualBlock

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # 输入z (B,4,H/8,W/8)
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            # (B,4,H/8,W/8) -> (B,512,H/8,W/8)
            nn.Conv2d(4,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            # (B,512,H/8,W/8) -> (B,512,H/4,W/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            # (B,512,H/4,W/4) -> (B,512,H/2,W/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            
            # (B,512,H/2,W/2) -> (B,256,H/2,W/2)
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            
            # (B,256,H/2,W/2) -> (B,256,H,W) 
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),

            # (B,256,H,W) ->  (B,128,H,W)
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32,128),
            nn.SiLU(),
            # (B,128,H,W) -> (B,3,H,W)
            nn.Conv2d(128,3,kernel_size=3,padding=1),
        )

    def forward(self, x):
        # x : (B,4,H/8,W/8) 来自encode解压的潜空间z
        # 移除encoder加的系数(sd源码)
        x /= 0.18215

        for module in self:
            x = module(x)
        
        # (B,3,H,W)
        return x;
