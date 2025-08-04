# CLIP的文本编码器
import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIP(nn.Module):
    '''    
            CLIP Embedding               12 CLIP Layers
    Tokens -----------------> Embedding ---------------> Output
                                          + LayerNorm
    '''
    def __init__(self):
        super().__init__()
        # 词表大小 = 49408
        self.embedding = CLIPEmbedding(49408,768,77)

        self.layers = nn.ModuleList([
            CLIPLayer(12,768) for i in range(12)
        ]) 

        self.layernorm = nn.LayerNorm(768)

    # emdedding 层只接受int64
    def forward(self,tokens : torch.LongTensor) -> torch.FloatTensor:
        
        tokens = tokens.type(torch.long)

        # (B,seq_len) -> (B,seq_len,d_embed)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        output = self.layernorm(state)

        return output
    
# 嵌入 + 位置编码
class CLIPEmbedding(nn.Module):
    # 词表大小 ; 嵌入的维度 ; Token序列长度
    def __init__(self, n_vocab : int, n_embd : int , n_token : int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab,n_embd)
        # 可学习的权重矩阵,用于进行位置编码
        self.position_embedding = nn.Parameter(torch.zeros((n_token,n_embd)))

    def forward(self,tokens):
        # tokens : (B,seq_len) 

        # (B,seq_len) -> (B,seq_len,d_embd)
        x = self.token_embedding(tokens)
        x += self.position_embedding

        return x

# 类似Transformer 
class CLIPLayer(nn.Module):
    def __init__(self,n_head : int,n_embd : int):
        super().__init__()

        # Attention Part
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention   = SelfAttention(n_head,n_embd)

        # FFN Part
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1    = nn.Linear(n_embd,4 * n_embd)
        self.linear_2    = nn.Linear(4 * n_embd,n_embd)

    def forward(self,x):
        # x : (B,seq_len,d_embed)
        residue = x

        # Attention Part
        x = self.layernorm_1(x)
        x = self.attention(x,causal_mask = True)
        x += residue

        #  FFN Part
        residue = x
        x = self.layernorm_2(x)
        # (B,seq_len,d_embed) -> (B,seq_len,4 * d_embed) 
        x = self.linear_1(x)
        # QuickGELU 激活函数
        x = x * torch.sigmoid(1.702 * x)
        # (B,seq_len,4 * d_embed) -> (B,seq_len,d_embed) 
        x = self.linear_2(x)

        x += residue

        return x
