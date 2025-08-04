# VAE Encoder
import torch
from torch import nn
from torch.nn import functional as F
from vae_blocks import VAE_AttentionBlock,VAE_ResidualBlock

# VAE 由卷积和注意力块构成
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # Input : (B,C=3,H,W) -> (B,128,H,W)
            nn.Conv2d(3,128,kernel_size = 3,padding = 1),

            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            # (B,128,H,W) -> (B,128,H/2,W/2)
            nn.Conv2d(128,128,kernel_size = 3,stride = 2 ,padding = 0),
           
            # (B,128,H/2,W/2) -> (B,256,H/2,W/2)
            VAE_ResidualBlock(128,256),   
            VAE_ResidualBlock(256,256),
            # (B,256,H/2,W/2) -> (B,256,H/4,W/4)
            nn.Conv2d(256,256,kernel_size = 3,stride = 2 ,padding = 0),
            
            # (B,256,H/4,W/4) -> (B,512,H/4,W/4)
            VAE_ResidualBlock(256,512),    
            VAE_ResidualBlock(512,512),
            # (B,512,H/4,W/4) -> (B,512,H/8,W/8)
            nn.Conv2d(512,512,kernel_size = 3,stride = 2 ,padding = 0),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            
            nn.GroupNorm(32,512),
            nn.SiLU(),
            # (B,512,H/8,W/8) -> (B,8,H/8,W/8)
            nn.Conv2d(512,8,kernel_size = 3,padding = 1),
            
            nn.Conv2d(8,8,kernel_size = 1,padding = 0),
        )
    '''
    VAE 公式:
    z = mean + sigma * noise  
    其中的noise与扩散的noise无关!
    '''
    def forward(self, x, noise):
        # x     : (B,C=3,H=512,W=512)   在diffusion中输入原图 
        # noise : (B,4,H/8,W/8)  与最终x维度一致

        # 按Sequential提供的模块计算
        for module in self:
            # 下采样的填充是非对称的（仅填充右侧，下侧）
            if getattr(module,'stride',None) == (2,2):
                x = F.pad(x,(0,1,0,1)) 
            x = module(x)

        # (B,8,H/8,W/8) -> 2 * (B,4,H/8,W/8)
        mean,log_variance = torch.trunk(x,2,dim = 1)

        log_variance = torch.clamp(log_variance,-30,20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # VAE 公式： N(0,1) -> N(mean,stdev)
        x = mean + stdev * noise
        # sd源码：https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215 

        return x