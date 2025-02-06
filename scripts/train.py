import torch
import torch.nn as nn
import torch.nn.functional as F
import json

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer.model import Transformer
from transformer.dataset import Seq2SeqDataset
from utils.process import data_process, generate_vocab


training_config = {
    'batch_size': 64,
    'epochs': 10,
    'lr': 1e-4,
    'num_layers': 3,
    'd_model': 512,
    'num_heads': 8,
    'd_k': 64,
    'd_ff': 2048,
    'dropout': 0.1,
    'max_len': 256,
    'pad_idx': 0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_path': 'model.pth',
    'load_path': 'model.pth',
    'src_language': 'chinese',
    'trg_language': 'english',
    'dataset_path': '../dataset/cn-en.json'
}

with open(training_config['dataset_path'], 'r') as f:
    data = json.load(f)
src = list(data['src'])
trg = list(data['trg'])



