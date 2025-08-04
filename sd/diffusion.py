# 包含隐空间中全部网络结构
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention
from unet_blocks import UNET_ResidualBlock,UNET_AttentionBlock,Upsample 

# 总体结构
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 对时间编码后传入UNet
        self.time_embedding = TimeEmbedding(320)
        # 处理潜空间图像与文本嵌入
        self.unet = UNET()
        # UNET输出与潜空间通道不匹配,需要额外转换
        self.final = UNET_OutputLayer(320,4)

    def forward(self,latent,context,time):
        # latent : (B, 4, H / 8, W / 8)
        # context: (B, seq_len , d_embed)
        # time   : (1, 320)

        # 时间向量映射
        # (1, 320) -> (1,1280)
        time = self.time_embedding(time)

        # (B, 4, H/8, W/8)   -> (B, 320, H/8, W/8)
        output = self.unet(latent,context,time)
        # (B, 320, H/8, W/8) -> (B, 4, H/8, W/8)
        output = self.final(output)

        return output
    
################  UNet相关结构 ######################

# 辅助模块,会根据当前序列的模块种类确定前向计算时的输入
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer,UNET_AttentionBlock):
                x = layer(x,context)
            elif isinstance(layer,UNET_ResidualBlock):
                x = layer(x,time)
            else:
                x = layer(x)
        return x

# 时间嵌入：(1,320) -> (1,1280)
class TimeEmbedding(nn.Module):
    def __init__(self,n_embed):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed,4 * n_embed)

    def forward(self,x):
        # x : (1 , 320)
        # (1,320) -> (1,1280)
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)

        return x
                

class UNET(nn.Module):
    def __init__(self):   
        super().__init__()

        self.encoders = nn.ModuleList([
            # (B,4,H/8,W/8) -> (B,320,H/8,W/8)
            SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock(8,40)),
            
            # (B,320,H/8,W/8)   -> (B,320,H/16,W/16)
            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
            # (B,320,H/16,W/16) -> (B,640,H/16,W/16)
            SwitchSequential(UNET_ResidualBlock(320,640),UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(640,640),UNET_AttentionBlock(8,80)),

            # (B,640,H/16,W/16) -> (B,640,H/32,W/32)
            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),
            # (B,640,H/32,W/32) -> (B,1280,H/32,W/32)
            SwitchSequential(UNET_ResidualBlock(640,1280),UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1280,1280),UNET_AttentionBlock(8,160)),

            # (B,1280,H/32,W/32) -> (B,1280,H/64,W/64)
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),
            
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
            SwitchSequential(UNET_ResidualBlock(1280,1280)),
        ])

        # (B,1280,H/64,W/64)
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280,1280),
            UNET_AttentionBlock(8,160),
            UNET_ResidualBlock(1280,1280),
        )

        # 注意decoder中ResBlock的输入维度由于拼接导致变化
        self.decoders = nn.ModuleList([
            # (B,1280+1280,H/64,W/64) -> (B,1280,H/64,W/64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            # (B,1280+1280,H/64,W/64) -> (B,1280,H/64,W/64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),
            # (B,1280+1280,H/64,W/64) -> (B,1280,H/64,W/64) -> (B,1280,H/32,W/32)
            SwitchSequential(UNET_ResidualBlock(2560,1280),Upsample(1280)),

            # (B,1280+1280,H/32,W/32) -> (B,1280,H/32,W/32)
            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            # (B,1280+1280,H/32,W/32) -> (B,1280,H/32,W/32)
            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),
            # (B,1280+640,H/32,W/32) -> (B,1280,H/32,W/32) -> (B,1280,H/16,W/16) 
            SwitchSequential(UNET_ResidualBlock(1920,1280),UNET_AttentionBlock(8,160),Upsample(1280)),

            # (B,1280+640,H/16,W/16) -> (B,640,H/16,W/16)
            SwitchSequential(UNET_ResidualBlock(1920,640),UNET_AttentionBlock(8,80)),
            # (B,640+640 ,H/16,W/16) -> (B,640,H/16,W/16)
            SwitchSequential(UNET_ResidualBlock(1280,640),UNET_AttentionBlock(8,80)),
            # (B,640+320, H/16,W/16) -> (B,640,H/16,W/16) -> (B,640,H/8,W/8)
            SwitchSequential(UNET_ResidualBlock(960,640) ,UNET_AttentionBlock(8,80),Upsample(640)),

            # (B,640+320,H/8,W/8)  -> (B,320,H/8,W/8)  
            SwitchSequential(UNET_ResidualBlock(960,320),UNET_AttentionBlock(8,40)),
            # (B,320+320,H/8,W/8)  -> (B,320,H/8,W/8)
            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,40)),
            # (B,320+320,H/8,W/8)  -> (B,320,H/8,W/8)
            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,40)),
        ])

    def forward(self,x,context,time):
        # x       : (B,4,H/8,W/8)
        # context : (B,seq_len,d_embed)
        # time    : (1,1280)

        # UNet跨层连接
        # 跨层连接的单位是SwitchSequential层
        skip_connections = []
        
        # UNet前向计算
        for layers in self.encoders:
            x = layers(x,context,time)
            # 逐步存储每次输出的特征
            skip_connections.append(x)

        x = self.bottleneck(x,context,time)

        for layers in self.decoders:
            # 跨层连接：维度拼接
            x = torch.cat((x,skip_connections.pop()),dim=1)
            x = layers(x,context,time)

        return x

# UNET 输出: (B,320,H/8,W/8) -> (B,4,H/8,W/8)
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32,in_channels)
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

    def forward(self,x):
        # x : (B, 320, H/8, W/8)
        x = self.groupnorm(x)
        x = F.silu(x)
        # (B, 320, H/8, W/8) -> (B, 4, H/8, W/8)
        x = self.conv(x)
        return x

