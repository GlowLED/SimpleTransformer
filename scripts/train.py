import torch
import torch.nn as nn
from tqdm import tqdm
import re
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from transformer.model import Transformer
from utils.process import prepare_for_train, generate_vocab_from_dataset, idx2tokens, detokenize, traditional_to_simplified

def read_data(file_path):
    src_texts = []
    trg_texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            src, trg = line.strip().split('\t')
            trg = traditional_to_simplified(trg)
            src_texts.append(src)
            trg_texts.append(trg)
    return src_texts, trg_texts

src_texts, trg_texts = read_data(r'data/train.txt')

class Dataset:
    def __init__(self, src_texts, trg_texts):
        self.src_texts = src_texts
        self.trg_texts = trg_texts

    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, index):
        return self.src_texts[index], self.trg_texts[index]


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_num = len(self.dataset) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset))
        else:
            indices = torch.arange(len(self.dataset))
        for i in range(self.batch_num):
            batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
            batch_src_texts = [self.dataset.src_texts[i] for i in batch_indices]
            batch_trg_texts = [self.dataset.trg_texts[i] for i in batch_indices]
            yield batch_src_texts, batch_trg_texts
    
    def __len__(self):
        return self.batch_num

src_texts, trg_texts = read_data(r'data/train.txt')
dataset = Dataset(src_texts, trg_texts)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
src_vocab, trg_vocab = generate_vocab_from_dataset(dataset, src_language='english', trg_language='chinese', max_vocab_size=10000)
de_trg_vocab = {idx: token for token, idx in trg_vocab.items()}
model = Transformer(
    d_model=1024,
    multihead_attn_h=64,
    num_heads=16,
    feedforward_h=2048,
    num_layers=8,
    src_vocab_size=len(src_vocab),
    trg_vocab_size=len(trg_vocab),
    max_len=1024,
    dropout=0.1
)
# warm up
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step_num: 512 ** (-0.5) * min((step_num + 1) ** (-0.5), (step_num + 1) * 4000 ** (-1.5)))
criterion = nn.CrossEntropyLoss(ignore_index=0)
def train(model, dataloader, optimizer, criterion, src_vocab, trg_vocab, epochs, device):
    
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for batch_src_texts, batch_trg_texts in tqdm(dataloader, total=len(dataloader)):
            batch_src, batch_trg_input, batch_trg_target = prepare_for_train(
                batch_src_texts, batch_trg_texts,
                src_vocab, trg_vocab,
                src_language='english', trg_language='chinese',
                bos_idx=1, eos_idx=2, pad_idx=0
            )

            batch_src, batch_trg_input, batch_trg_target = batch_src.to(device), batch_trg_input.to(device), batch_trg_target.to(device)
            optimizer.zero_grad()
            output = model(batch_src, batch_trg_input)
            loss = criterion(output.view(-1, output.shape[-1]), batch_trg_target.view(-1))
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        print(batch_src_texts[0])
        print(batch_trg_texts[0])
        trg_tokens = idx2tokens(output[0].argmax(dim=-1).tolist(), de_trg_vocab)
        print(detokenize(trg_tokens, 'chinese'))
        
    return model

model = train(
    model, dataloader, optimizer, criterion, src_vocab, trg_vocab, epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu'
)

# save model
torch.save(model.state_dict(), 'model.pth')

# save vocab
import pickle
with open('src_vocab.pkl', 'wb') as f:
    pickle.dump(src_vocab, f)
with open('trg_vocab.pkl', 'wb') as f:
    pickle.dump(trg_vocab, f)
