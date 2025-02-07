import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.mask_generate import padding_mask, look_ahead_mask
from transformer.module import Encoder, Decoder, PositionalEncoding


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        input_vocab_size,
        target_vocab_size,
        max_len=2048,
        dropout=0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.dropout = dropout
        self.max_len = max_len
        
        self.input_emb = nn.Embedding(input_vocab_size, d_model)
        self.output_emb = nn.Embedding(target_vocab_size, d_model)
        
        self.pos_enc = PositionalEncoding(
            d_model=d_model,
            max_len=max_len
        )
        
        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        
        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
    
        self.output_linear = nn.Linear(d_model, target_vocab_size)
    
    def forward(self, src, trg):
        '''
        Args:
            src: [batch_size, src_len]
            trg: [batch_size, trg_len]
            src_padding_mask: [batch_size, src_len, src_len]
            trg_padding_mask: [batch_size, trg_len, trg_len]
            trg_look_ahead_mask: [batch_size, trg_len, trg_len]
        Returns:
            logits: [batch_size, trg_len, target_vocab_size]
        '''
        
        src_emb = self.input_emb(src)
        trg_emb = self.output_emb(trg)
        
        src_emb = self.pos_enc(src_emb)
        trg_emb = self.pos_enc(trg_emb)
        
        enc_self_attn_mask = padding_mask(src, src)
        dec_self_attn_mask = padding_mask(trg, trg) & look_ahead_mask(trg)
        dec_cross_attn_mask = padding_mask(trg, src)
        
        enc_output = self.encoder(src_emb, enc_self_attn_mask)
        logits = self.decoder(trg_emb, enc_output, dec_self_attn_mask, dec_cross_attn_mask)
        
        logits = self.output_linear(logits)
        logits = F.softmax(logits, dim=-1)
        
        return logits
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f'model saved at {path}')
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f'model loaded from {path}')
        return self  
    
    def to(self, device):
        super().to(device)
        self.input_emb.to(device)
        self.output_emb.to(device)
        self.pos_enc.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.output_linear.to(device)
        return self
    
      

    
