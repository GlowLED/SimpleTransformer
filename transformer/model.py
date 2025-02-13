import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.mask_generate import padding_mask, look_ahead_mask
from transformer.module import Embedding, PositionalEncoding, Encoder, Decoder

class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        multihead_attn_h,
        num_heads,
        feedforward_h,
        num_layers,
        src_vocab_size,
        trg_vocab_size,
        max_len=1024,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.multihead_attn_h = multihead_attn_h
        self.num_heads = num_heads
        self.feedforward_h = feedforward_h
        self.num_layers = num_layers
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.max_len = max_len
        self.dropout = dropout
        
        self.src_emb = Embedding(src_vocab_size, d_model)
        self.trg_emb = Embedding(trg_vocab_size, d_model)
        
        self.pos_enc = PositionalEncoding(d_model, max_len)
        
        self.encoder = Encoder(
            d_model,
            multihead_attn_h,
            num_heads,
            feedforward_h,
            num_layers,
            dropout
        )
        
        self.decoder = Decoder(
            d_model,
            multihead_attn_h,
            num_heads,
            feedforward_h,
            num_layers,
            dropout
        )
        
        self.linear = nn.Linear(d_model, trg_vocab_size)
        
    
    def forward(self, src, trg, pad_idx=0):
        '''
        Args:
            src: (batch_size, src_seq_len)
            trg: (batch_size, trg_seq_len)
        Returns:
            o: (batch_size, trg_seq_len, trg_vocab_size)
        '''
         
        src_emb = self.src_emb(src)
        trg_emb = self.trg_emb(trg)
        
        src_emb = self.pos_enc(src_emb)
        trg_emb = self.pos_enc(trg_emb)
        
        enc_self_attn_mask = padding_mask(src, src, pad_idx)
        dec_self_attn_mask = torch.logical_or(padding_mask(trg, trg, pad_idx), look_ahead_mask(trg))
        dec_cross_attn_mask = padding_mask(trg, src, pad_idx)

        
        enc_o = self.encoder(src_emb, enc_self_attn_mask)
        dec_o = self.decoder(trg_emb, enc_o, dec_self_attn_mask, dec_cross_attn_mask)
        
        o = self.linear(dec_o)
        
        return o
    

    


    