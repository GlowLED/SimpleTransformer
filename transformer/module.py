import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        '''
        pe: [1, max_len, d_model]
        '''
        
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        '''
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        '''

        x = x + self.pe[:, :x.size(1), :]
        
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dk):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.sqrt_dk = math.sqrt(dk)
        
        self.Wq = nn.Linear(d_model, num_heads*dk, bias=False)
        self.Wk = nn.Linear(d_model, num_heads*dk, bias=False)
        self.Wv = nn.Linear(d_model, num_heads*dk, bias=False)
        self.Wo = nn.Linear(num_heads*dk, d_model, bias=False)
    
    def forward(self, xq, xk, xv, mask=None):
        '''
        Args:
            xq: [batch_size, seq_len, d_model]
            xk: [batch_size, seq_len, d_model]
            xv: [batch_size, seq_len, d_model]
            mask: a mask for attention, [batch_size, seq_len, seq_len]
        
        Returns:
            x: [batch_size, seq_len, d_model]
        '''
        
        q = self.Wq(xq)
        k = self.Wk(xk)
        v = self.Wv(xv)
        
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_dk
        
        if mask is not None:
            attn_score = attn_score.masked_fill(~mask, -1e9)
        
        attn_score = F.softmax(attn_score, dim=-1)
        
        x = torch.matmul(attn_score, v)
        x = self.Wo(x)
        
        return x
        
        
class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, y):
        '''
        Args:
            x: initial input, [batch_size, seq_len, d_model]
            y: attn output or feedforward output, [batch_size, seq_len, d_model]
        Returns:
            x: the result of add and norm, [batch_size, seq_len, d_model]
        '''
        
        x = self.norm(x + y)
        
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        '''
        Args:
            x: attn output, [batch_size, seq_len, d_model]
        Returns:
            x: feedforward output, [batch_size, seq_len, d_model]
        '''
                             
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x

class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        

        self.multi_head_attns = nn.ModuleList(
            [MultiHeadAttention(d_model, num_heads, d_model//num_heads) for _ in range(num_layers)]
        )
        self.addnorms1 = nn.ModuleList(
            [AddNorm(d_model) for _ in range(num_layers)]
        )
        self.feedforwards = nn.ModuleList(
            [FeedForward(d_model, d_ff, dropout) for _ in range(num_layers)]
        )
        self.addnorms2 = nn.ModuleList(
            [AddNorm(d_model) for _ in range(num_layers)]
        )
    
    def forward(self, x, self_attn_mask):
        '''
        Args:
            x: [batch_size, seq_len, d_model]
            padding_mask: [batch_size, seq_len, seq_len]
        Returns:
            x: [batch_size, seq_len, d_model]
        '''
        
        for i in range(self.num_layers):
            y = self.multi_head_attns[i](x, x, x, self_attn_mask)
            x = self.addnorms1[i](x, y)
            y = self.feedforwards[i](x)
            x = self.addnorms2[i](x, y)
        
        return x
    

class Decoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        self.masked_multi_head_attns = nn.ModuleList(
            [MultiHeadAttention(d_model, num_heads, d_model//num_heads) for _ in range(num_layers)]
        )
        
        self.addnorms1 = nn.ModuleList(
            [AddNorm(d_model) for _ in range(num_layers)]
        )
        
        self.multi_head_attns = nn.ModuleList(
            [MultiHeadAttention(d_model, num_heads, d_model//num_heads) for _ in range(num_layers)]
        )

        self.addnorms2 = nn.ModuleList(
            [AddNorm(d_model) for _ in range(num_layers)]
        )
        
        self.feedforwards = nn.ModuleList(
            [FeedForward(d_model, d_ff, dropout) for _ in range(num_layers)]
        )
        
        self.addnorms3 = nn.ModuleList(
            [AddNorm(d_model) for _ in range(num_layers)]
        )
        
    
    def forward(self, x, enc_output, self_attn_mask, cross_attn_mask):
        '''
        Args:
            x: [batch_size, seq_len, d_model]
            padding_mask: [batch_size, seq_len, seq_len]
            look_ahead_mask: [batch_size, seq_len, seq_len]
        Returns:
            x: [batch_size, seq_len, d_model]
        '''

        
        for i in range(self.num_layers):
            y = self.masked_multi_head_attns[i](x, x, x, self_attn_mask)
            x = self.addnorms1[i](x, y)
            y = self.multi_head_attns[i](x, enc_output, enc_output, cross_attn_mask)
            x = self.addnorms2[i](x, y)
            y = self.feedforwards[i](x)
            x = self.addnorms3[i](x, y)
        
        return x
            


