# 包含隐空间中全部网络结构
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

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