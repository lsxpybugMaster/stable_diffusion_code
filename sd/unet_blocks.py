import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention,CrossAttention

class Upsample(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        # (B,C,H,W) ->  (B,C,H*2,W*2)
        x = F.interpolate(x,scale_factor=2,mode='nearest')
        return self.conv(x)
    
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,n_time = 1280):
        super().__init__()
        # 特征图管线
        self.groupnorm_feature = nn.GroupNorm(32,in_channels);
        self.conv_feature = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        # 文本管线
        self.linear_time = nn.Linear(n_time,out_channels)
        # 合并文本与特征图
        self.groupnorm_merged = nn.GroupNorm(32,out_channels)
        self.conv_merged = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0)
    
    def forward(self,feature,time):
        # x : (B, in_channels, H, W)
        # time : (1,1280)

        residue = feature

        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        # (B, in_channels, H, W) -> (B, out_channels, H, W)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        # (1,1280) -> (1, out_channels)
        time = self.linear_time(time)

        # 合并两个支路
        # (B, out_channels, H, W) + (1, out_channels, 1, 1) => (B, out_channels, H, W)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head:int ,n_embd:int, d_context = 768):
        super().__init__()

        channels = n_head * n_embd

        # 将特征图reshape成序列特征
        self.groupnorm = nn.GroupNorm(32,channels,eps = 1e-6)
        self.conv_input = nn.Conv2d(channels,channels,kernel_size=1,padding=0)
        # Transformer结构
        # Self-Attn
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head,channels,in_proj_bias=False)
        # Cross-Attn
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head,channels,d_context,in_proj_bias=False)
        # FFN
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels,4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self,x,context):
        # x : (B, C, H, W)
        # context : (B, seq_len , d_embed)

        residue_long = x

        # 预处理特征图维度
        x = self.groupnorm(x) 
        x = self.conv_input(x)
        n, c, h, w = x.shape
        # (B, C, H, W) -> (B, C, H * W) -> (B, H * W, C)
        x = x.view((n,c,h * w))
        x = x.transpose(-1,-2)

        # ----------------------Self-Attn--------------------------
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # ----------------------Cross-Attn--------------------------
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x,context)
        x += residue_short

        # ----------------------FFN(GeGLU)--------------------------
        residue_short = x
        x = self.layernorm_3(x)
        # GeGLU https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (B, H * W, C) -> 2 * (B, H * W, C * 4)
        x, gate = self.linear_geglu_1(x).chunk(2,dim = -1)
        # (B, H * W, C * 4) * (B, H * W, C * 4) 逐个元素乘
        x = x * F.gelu(gate)
        # (B, H * W, C * 4) -> (B, H * W, C) 
        x = self.linear_geglu_2(x)

        x += residue_short

        # 还原维度
        x = x.transpose(-1,-2)
        x = x.view((n,c,h,w))

        return self.conv_output(x) + residue_long

