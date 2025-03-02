import torch
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from transformer.model import Transformer

def inference(model, src, bos_idx=1, eos_idx=2, max_len=1024):
    trg = list(bos_idx)
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg).unsqueeze(0)
        with torch.no_grad():
            enc_o = model.encoder(src)
            
        
