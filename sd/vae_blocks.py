import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention
'''
定义了VAE的两个模块:
AttentionBlock
ResdiualBlock
'''

# 区别继承nn.Module与nn.Sequential的网络定义方法
class VAE_AttentionBlock(nn.Module):
    def __init__(self, seq_len):
        super.__init__()

        
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()

        self.groupnorm_1 = nn.GroupNorm(32,in_channels)
        self.conv_1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

        self.groupnorm_2 = nn.GroupNorm(32,out_channels)
        self.conv_2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        # 如果输入输出通道不同,Res操作无法进行,此时需要将原始x用1*1卷积使其通道与输出对齐
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    
    def forward(self,x):
        # x : (B,in_channels,H,W)
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        # (B,in_channels,H,W) ->  (B,out_channels,H,W)
        x = self.conv_1(x)


        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        # residual_layer 一定会将residue的通道与输出对齐
        return x + self.residual_layer(residue)

       
