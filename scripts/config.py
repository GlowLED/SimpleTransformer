from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 512
    multihead_attn_h: int = 64
    num_heads: int = 8
    feedforward_h: int = 2048
    num_layers: int = 6
    src_vocab_size: int = 10000
    trg_vocab_size: int = 10000
    max_len: int = 1024
    
