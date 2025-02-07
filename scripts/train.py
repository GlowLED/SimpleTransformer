import torch
import json

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.process import data_process, load_vocab, generate_vocab

model_config = {
    'd_model': 512,
    'd_ff': 2048,
    'd_k': 64,
    'n_heads': 8,
    'n_encoder_layers': 6,
    'n_decoder_layers': 6,
    'dropout': 0.1,
    'max_len': 512,
    'src_vocab_size': 10000,    
    'trg_vocab_size': 10000,
}

training_config = {
    'batch_size': 32,
    'epochs': 100,
    'lr': 0.001,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'train_data_path': r'.data/train/',
    'eval_data_path': r'.data/eval/',
    'experiment_name': 'test',
    'experiment_directory': r'.experiments/',
    'load_model_path': r'experiments/test/model.pth',
    'log_interval': 100,
    'save_interval': 10000,
}

# create experiment directory
if not os.path.exists(training_config['experiment_directory']):
    os.makedirs(training_config['experiment_directory'])
os.makedirs(os.path.join(training_config['experiment_directory'], training_config['experiment_name']))


# get all file name in the data directory
for file_name in os.listdir(training_config['train_data_path']):
    ...
        
