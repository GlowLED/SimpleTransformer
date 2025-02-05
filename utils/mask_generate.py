import torch
import torch.nn.functional as F




def look_ahead_mask(seq):
    '''
    Args:
        seq: [batch_size, seq_len]
    
    Returns:
        mask: [batch_size, seq_len, seq_len]
    '''
    mask = ~torch.triu(torch.ones(seq.size(1), seq.size(1), device=seq.device), diagonal=1).bool()
    return mask


def padding_mask(seq_q, seq_k):
    '''
    Args:
        seq_q: [batch_size, seq_len_q]
        seq_k: [batch_size, seq_len_k]
    
    Returns:
        mask: [batch_size, seq_len_q, seq_len_k]
    '''
    mask_q = ~seq_q.eq(0).unsqueeze(-1)
    mask_k = ~seq_k.eq(0).unsqueeze(-2)
    mask = torch.bmm(mask_q.float(), mask_k.float()).bool()
    return mask
