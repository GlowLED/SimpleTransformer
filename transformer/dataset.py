import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Seq2SeqDataset(Dataset):
    def __init__(self, src, trg):
        super().__init__()
        self.src = src
        self.trg = trg

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):
        return self.src[index], self.trg[index]
    





    