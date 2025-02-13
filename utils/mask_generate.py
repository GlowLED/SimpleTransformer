'''
some mask generation functions

'''

import torch
import torch.nn.functional as F

def look_ahead_mask(seq):
    '''
    Args:
        seq: [batch_size, seq_len]
    
    Returns:
        mask: [seq_len, seq_len]
    '''
    
    # example:
    # False True True
    # False False True
    # False False False
    
    mask = torch.triu(torch.ones(seq.size(1), seq.size(1), device=seq.device), diagonal=1).bool()
    return mask

def padding_mask(seq_q, seq_k, pad_idx=0):
    '''
    Args:
        seq_q: [batch_size, seq_len_q]
        seq_k: [batch_size, seq_len_k]
    
    Returns:
        mask: [batch_size, seq_len_q, seq_len_k]
    '''
    
    # example:
    # False False False True
    # False False False True
    # False False False True
    
    mask = seq_k.eq(pad_idx).unsqueeze(1).repeat(1, seq_q.size(1), 1)
    return mask


