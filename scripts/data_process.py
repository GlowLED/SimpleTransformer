import pickle, json

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.process import tokenize
from tqdm import tqdm

path = r'.data/train'
file_list = os.listdir(path)
src = []
trg = []
for file in file_list:
    print(f'loading {file}')
    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for d in data:
            src.append(d['english'])
            trg.append(d['chinese'])

trg_tokens_list = [['<sos>'] + tokenize(text, language='chinese') + ['<eos>'] for text in tqdm(src, desc='trg')]
# save the list
save_path = r'.data/train_tokenized_data_trg.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(trg_tokens_list, f)
