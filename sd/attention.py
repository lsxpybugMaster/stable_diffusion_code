# 自注意力机制与交叉注意力机制
import torch
from torch import nn
from torch.nn import functional as F
import math

'''
注意力机制(多头) d_embed = D_EMBED / nheads
x            : (seq_len * D_EMBED)
Wq,Wk,Wv     : (D_EMBED * D_EMBED)
[Wq,Wk,Wv]   : (D_EMBED * 3D_EMBED)
Q = x @ Wq   : (seq_len * nheads * d_embed) K,V同理 
至此对每个单头 Q' K' V' 
K'_T         : (d_embed * seq_len)
Q' @ K'_T    : (seq_len * seq_len)
Sm(Q'K'/d)@V': (seq_len * d_embed)
Concat (C)   : (seq_len * D_EMBED)     
Wo           : (D_EMBED * D_EMBED)
y = C @ Wo   : (seq_len * D_EMBED)
'''

# 多头自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed,in_proj_bias = True,out_proj_bias = True):
        super().__init__()

