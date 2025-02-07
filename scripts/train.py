import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer.model import Transformer
from transformer.dataset import Seq2SeqDataset
from utils.process import data_process, generate_vocab, padding

# define the training config.
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

# load data.
with open(training_config['dataset_path'], 'r') as f:
    data = json.load(f)

# transform the json data into list of str.
src = list(data['src'])
trg = list(data['trg'])

# generate the vocab.
src_vocab = generate_vocab(src, language=training_config['src_language'])
trg_vocab = generate_vocab(trg, language=training_config['trg_language'])

# data process: transform the src and trg texts into idx.
src, trg = data_process(src, trg, src_vocab, trg_vocab, src_language=training_config['src_language'], trg_language=training_config['trg_language'], max_len=training_config['max_len'], pad_idx=training_config['pad_idx'])

# build dataset and dataloader.
dataset = Seq2SeqDataset(src, trg)
dataloader = DataLoader(dataset, batch_size=training_config['batch_size'], shuffle=True)

# create the model.
model = Transformer(
    num_layers=training_config['num_layers'],
    d_model=training_config['d_model'],
    num_heads=training_config['num_heads'],
    d_ff=training_config['d_ff'],
    input_vocab_size=len(src_vocab),
    target_vocab_size=len(trg_vocab),
    max_len=training_config['max_len'],
)

# load the model.
if os.path.exists(training_config['load_path']):
    model.load(training_config['load_path'])
    print('load model from {}'.format(training_config['load_path']))
else:
    print('no model found, start training from scratch.')

# move the model to device.
model.to(training_config['device'])

# create the optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=training_config['lr'])

# create the scheduler.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# define the loss function.
loss_fn = nn.CrossEntropyLoss(ignore_index=training_config['pad_idx'])

# start training.
for epoch in range(training_config['epochs']):
    model.train()
    for batch in dataloader:
        src, trg = batch
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:] # (batch_size, trg_len)
        src, trg_input = padding(src, max_len=training_config['max_len'], pad_idx=training_config['pad_idx']),\
            padding(trg_input, max_len=training_config['max_len'], pad_idx=training_config['pad_idx'])
        trg_output = padding(trg_output, max_len=training_config['max_len'], pad_idx=training_config['pad_idx'])
        trg_output_one_hot = F.one_hot(trg_output, num_classes=len(trg_vocab)).float() # (batch_size, trg_len, trg_vocab_size)
        
        # move data to device.
        src = src.to(training_config['device'])
        trg_input = trg_input.to(training_config['device'])
        trg_output_one_hot = trg_output_one_hot.to(training_config['device'])
        
        # predict.
        pred = model(src, trg_input)
        
        # calculate loss.
        loss = loss_fn(pred, trg_output_one_hot)
        
        # backward.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
    print('epoch: {}, loss: {:.4f}'.format(epoch, loss.item()))
    if epoch % 5 == 0:
        model.save(training_config['save_path'])
        print('save model to {}'.format(training_config['save_path']))

        
        
        
