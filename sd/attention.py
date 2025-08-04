# 自注意力机制与交叉注意力机制
import torch
from torch import nn
from torch.nn import functional as F
import math

'''
注意力机制(多头) d_head = d_embed / nheads
x            : (seq_len * d_embed)
Wq,Wk,Wv     : (d_embed * d_embed)
[Wq,Wk,Wv]   : (d_embed * 3d_embed)
Q = x @ Wq   : (seq_len * nheads * d_head) K,V同理 
至此对每个单头 Q' K' V' 
K'_T         : (d_head  * seq_len)
Q' @ K'_T    : (seq_len * seq_len)
Sm(Q'K'/d)@V': (seq_len * d_head )
Concat (C)   : (seq_len * d_embed)     
Wo           : (d_embed * d_embed)
y = C @ Wo   : (seq_len * d_embed)
'''

# 多头自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias = True,out_proj_bias = True):
        super().__init__()

        # [Wq,Wk,Wv] 合并矩阵，一次计算
        self.in_proj = nn.Linear(d_embed,3*(d_embed),bias=in_proj_bias)
        # Wo
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bias)
        
        # 记录多头相关信息 
        self.n_heads = n_heads
        # 记录每一个头的embed维度
        self.d_head = d_embed // n_heads

    # 注意自注意力机制有Mask
    def forward(self,x,causal_mask = False):
        # x : (B,seq_len,d_embed)

        # 提前存取输入形状信息
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # 提前定义多头形状信息(多了一个表示头的维度)
        interim_shape = (batch_size,sequence_length,self.n_heads,self.d_head)

        # (B,seq_len,d_embed) -> (B,seq_len,3 * d_embed) -> 3 * (B,seq_len,d_embed)
        q, k, v = self.in_proj(x).chunk(3,dim = -1)

        # (B,seq_len,d_embed) -> (B,seq_len,n_heads,d_head) 切分多头
        # (B,seq_len,n_heads,d_head) -> (B,n_heads,seq_len,d_head) 更换维度顺序
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        # '@'自动计算后两维矩阵乘法
        # (B,n_heads,seq_len,d_head) @ (B,n_heads,d_head,seq_len) = (B,n_heads,seq_len,seq_len)
        weight = q @ k.transpose(-1,-2)

        # 是否增加Mask
        if causal_mask:
            # 创建全1三角矩阵(右上角不含对角)并掩码weight
            mask = torch.ones_like(weight,dtype=torch.bool).triu(1)
            weight.masked_fill_(mask,-torch.inf)

        # 除sqrt(d_head)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim = -1)
        # (B,n_heads,seq_len,seq_len) @ (B,n_heads,seq_len,d_head) = (B,n_heads,seq_len,d_head)
        output = weight @ v;

        # 通过重置维度拼接各个头的输出
        # (B,n_heads,seq_len,d_head) -> (B,seq_len,n_heads,d_head)
        output = output.transpose(1,2)
        # (B,seq_len,n_heads,d_head) -> (B,seq_len,d_embed)
        output = output.reshape(input_shape)

        # 拼好的头乘Wo
        output = self.out_proj(output)

        return output
    
class CrossAttention(nn.Module):
    def __init__(self,n_heads,d_embed,d_cross,in_proj_bias = True,out_proj_bias = True):
        super().__init__()
        
        self.q_proj   = nn.Linear(d_embed,d_embed,bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross,d_embed,bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross,d_embed,bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed,d_embed,bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self,x,y):
        # x (latent) : (B,seq_len_q ,dim_q )
        # y (context): (B,seq_len_kv,dim_kv) = (B,77,768)
        
        input_shape = x.shape
        batch_size,sequence_length,d_embed = input_shape
        # 序列长度不同，不能直接定义seq_len值
        interim_shape = (batch_size,-1,self.n_heads,self.d_head)

        q = self.q_proj(x) # (B,seq_len_q,dim_q)
        k = self.k_proj(y) # (B,seq_len_kv,dim_kv)->(B,seq_len_kv,dim_q)
        v = self.v_proj(y) # (B,seq_len_kv,dim_kv)->(B,seq_len_kv,dim_q)

        # (B,seq_len_q,dim_q) -> (B,seq_len_q,n_head,d_head) -> (B,n_head,seq_len_q,d_head)
        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)

        weight = q @ k.transpose(-1,-2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight,dim=-1)
        output = weight @ v

        # 拼接多头  
        # (B,n_head,seq_len_q,d_head) -> (B,seq_len_q,n_head,d_head)
        output = output.transpose(1,2).contiguous()
        # (B,seq_len_q,n_head,d_head) -> (B,seq_len_q,dim_q)
        output = output.view(input_shape)

        output = self.out_proj(output)

        return output