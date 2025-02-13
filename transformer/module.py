import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len):
        super().__init__()
        self.d = d
        self.max_len = max_len

        pe = torch.zeros((1, max_len, d))
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        x = x / torch.pow(10000, torch.arange(0, d, 2, dtype=torch.float32) / d)

        pe[:, :, 0::2] = torch.sin(x)
        pe[:, :, 1::2] = torch.cos(x)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        Args:
            x: [batch_size, seq_len, d]
        Returns:
            o: [batch_size, seq_len, d]
        '''
        
        o = x + self.pe[:, :x.shape[1], :]
        
        return o
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_xq, d_xk, d_xv, h, num_heads, dropout=0.1):
        '''
        Args:
            d_xq: dimension of query
            d_xk: dimension of key
            d_xv: dimension of value
            h: dimension of hidden
            num_heads: number of heads
            dropout: dropout rate
        '''

        super().__init__()
        self.d_xq = d_xq
        self.d_xk = d_xk
        self.d_xv = d_xv
        self.num_heads = num_heads
        self.h = h

        self.W_q = nn.Linear(d_xq, h * num_heads, bias=False)
        self.W_k = nn.Linear(d_xk, h * num_heads, bias=False)
        self.W_v = nn.Linear(d_xv, h * num_heads, bias=False)
        self.W_o = nn.Linear(h * num_heads, d_xv, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, xq, xk, xv, mask=None):
        '''

        Args:
            xq: query, [batch_size, n_q, d_xq]
            xk: key, [batch_size, n_k, d_xk]
            xv: value, [batch_size, n_v, d_xv]
            mask: [batch_size, n_q, n_k] or [n_q, n_k] or None, True: mask, False: no mask
 
        Returns:
            o: [batch_size, n_q, d_xv]

        '''
        
        # get batch_size, n_q, n_k, n_v, num_heads, h.
        n_q, n_k, n_v = xq.size(1), xk.size(1), xv.size(1)
        num_heads = self.num_heads
        h = self.h
        # separate each head.
        # (batch_size, n, h * num_heads) -> (batch_size, n, num_heads, h) -> (batch_size, num_heads, n, h) -> (batch_size * num_heads, n, h)
        q = self.W_q(xq).reshape(-1, n_q, num_heads, h).permute(0, 2, 1, 3).reshape(-1, n_q, h)
        k = self.W_k(xk).reshape(-1, n_k, num_heads, h).permute(0, 2, 1, 3).reshape(-1, n_k, h)
        v = self.W_v(xv).reshape(-1, n_v, num_heads, h).permute(0, 2, 1, 3).reshape(-1, n_v, h)

        # Q*K^T/sqrt(h)
        attn_score = torch.bmm(q, k.transpose(-1, -2)) / math.sqrt(h)  # (batch_size * num_heads, n_q, n_k)

        # apply mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            elif mask.dim() == 3:
                mask = torch.repeat_interleave(mask, num_heads, dim=0)
            else:
                raise ValueError('mask dim must be 2 or 3')
            attn_score = attn_score.masked_fill(mask, -1e9)

        # softmax attn_score to get attn_weights, and apply it to v to get v_pool.
        attn_weights = F.softmax(attn_score, dim=-1)
        v_pool = torch.bmm(self.dropout(attn_weights), v)

        # (batch_size * num_heads * n_q, h) -> (batch_size, num_heads, n_q, h) -> (batch_size, n_q, num_heads, h) -> (batch_size, n_q, h * num_heads)
        v_pool = v_pool.reshape(-1, num_heads, n_q, h).permute(0, 2, 1, 3).reshape(-1, n_q, h * num_heads)
        
        o = self.W_o(v_pool)    # (batch_size, n_q, d_xv)
        return o


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_emb):
        super().__init__()
        self.d_emb = d_emb
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_emb)

    def forward(self, x):
        '''
        Args:
            x: [batch_size, seq_len]
        Returns:
            o: [batch_size, seq_len, d_emb]
        '''
        
        o = self.embedding(x)
        
        return o


class AddNorm(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.norm = nn.LayerNorm(d)
    
    def forward(self, x, y):
        '''
        Args:
            x: initial input, [batch_size, seq_len, d]
            y: attn output or feedforward output, [batch_size, seq_len, d]
        Returns:
            x: the result of add and norm, [batch_size, seq_len, d]
        '''

        x = self.norm(x + y)

        return x


class FeedForward(nn.Module):
    def __init__(self, d_i, d_h, d_o, dropout=0.1):
        super().__init__()
        self.d_i = d_i
        self.d_h = d_h
        self.d_o = d_o

        self.linear1 = nn.Linear(d_i, d_h)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_h, d_o)


    def forward(self, x):
        '''
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            o: [batch_size, seq_len, d_model]
        '''

        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        o = self.linear2(x)

        return o
    

class Encoder(nn.Module):
    def __init__(self, d_model, multihead_attn_h, num_heads, feedforward_h, num_layers, dropout=0.1):
        '''
        Args:
            d_model: dimension of model
            multihead_attn_h: dimension of hidden in multihead attention
            num_heads: number of heads
            feedforward_h: dimension of hidden in feedforward
            num_layers: number of layers
            dropout: dropout rate
        ''' 
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.multihead_attn_h = multihead_attn_h
        self.feedforward_h = feedforward_h
        self.dropout = dropout
        
        self.multihead_attns = nn.ModuleList([MultiHeadAttention(d_model, d_model, d_model, multihead_attn_h, num_heads, dropout) for _ in range(num_layers)])
        self.addnorms1 = nn.ModuleList([AddNorm(d_model) for _ in range(num_layers)])
        self.feedforwards = nn.ModuleList([FeedForward(d_model, feedforward_h, d_model, dropout) for _ in range(num_layers)])
        self.addnorms2 = nn.ModuleList([AddNorm(d_model) for _ in range(num_layers)])
    
    def forward(self, x, attn_mask):
        '''
        Args: 
            x: [batch_size, seq_len, d_model], embedding of the input
            attn_mask: [batch_size, seq_len, seq_len]
        Returns:
            x: [batch_size, seq_len, d_model]
        '''
        for i in range(self.num_layers):
            y = self.multihead_attns[i](x, x, x, attn_mask)
            x = self.addnorms1[i](x, y)
            y = self.feedforwards[i](x)
            x = self.addnorms2[i](x, y)
        
        return x
        
        
class Decoder(nn.Module):
    def __init__(self, d_model, multihead_attn_h, num_heads, feedforward_h, num_layers, dropout=0.1):
        '''
        Args:
            d_model: dimension of model
            multihead_attn_h: dimension of hidden in multihead attention
            num_heads: number of heads
            feedforward_h: dimension of hidden in feedforward
            num_layers: number of layers
            dropout: dropout rate
        '''
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.multihead_attn_h = multihead_attn_h
        self.feedforward_h = feedforward_h
        
        self.self_multihead_attns = nn.ModuleList([MultiHeadAttention(d_model, d_model, d_model, multihead_attn_h, num_heads, dropout) for _ in range(num_layers)])
        self.addnorms1 = nn.ModuleList([AddNorm(d_model) for _ in range(num_layers)])
        self.cross_multihead_attns = nn.ModuleList([MultiHeadAttention(d_model, d_model, d_model, multihead_attn_h, num_heads, dropout) for _ in range(num_layers)])
        self.addnorms2 = nn.ModuleList([AddNorm(d_model) for _ in range(num_layers)])
        self.feedforwards = nn.ModuleList([FeedForward(d_model, feedforward_h, d_model, dropout) for _ in range(num_layers)])
        self.addnorms3 = nn.ModuleList([AddNorm(d_model) for _ in range(num_layers)])
        
    def forward(self, x, enc_o, self_attn_mask, cross_attn_mask):
        '''
        Args:
            x: [batch_size, seq_len, d_model], embedding of the input
            enc_o: [batch_size, seq_len, d_model], output of encoder
            self_attn_mask: [batch_size, seq_len, seq_len]
            cross_attn_mask: [batch_size, seq_len, seq_len]
        Returns:
            x: [batch_size, seq_len, d_model]
        '''
        
        for i in range(self.num_layers):
            y = self.self_multihead_attns[i](x, x, x, self_attn_mask)
            x = self.addnorms1[i](x, y)
            y = self.cross_multihead_attns[i](x, enc_o, enc_o, cross_attn_mask)
            x = self.addnorms2[i](x, y)
            y = self.feedforwards[i](x)
            x = self.addnorms3[i](x, y)
        return x


    