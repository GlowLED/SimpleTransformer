'''
some functions to process text data
for example, tokenize, detokenize, generate_vocab, data_process

'''


import torch
import torch.nn.functional as F
import nltk
import jieba
import logging
import json, os
jieba.setLogLevel(logging.INFO)


def tokenize(text, language='english') -> list:
    if language == 'chinese':
        return list(jieba.cut(text))
    return nltk.word_tokenize(text, language=language)

def detokenize(tokens) -> str:
    return ' '.join(tokens)

def tokens2idx(tokens, en_vocab) -> torch.LongTensor:
    '''
    Args:
        tokens: list of str
        en_vocab: dict, {token: idx}
    Returns:
        longtensor of idx
    special tokens: <pad>, <sos>, <eos>, <unk>
    '''
    idx = torch.LongTensor([en_vocab[token] if token in en_vocab else en_vocab['<unk>'] for token in tokens])
    return idx

def idx2tokens(idx, de_vocab) -> list:
    '''
    Args:
        idx: longtensor
        de_vocab: dict, {idx: token}
    Returns:
        list of str
    '''
    tokens = [de_vocab[i.item()] for i in idx]
    return tokens

def padding(seqs, max_len, pad_idx=0) -> torch.LongTensor:
    '''
    Args:
        seqs: list of longtensor
        max_len: int
    Returns:
        longtensor of [len(seqs), max_len]
    '''
    batch_size = len(seqs)
    padded_seqs = torch.zeros(batch_size, max_len).long()
    for i, seq in enumerate(seqs):
        if len(seq) > max_len:
            padded_seqs[i] = seq[:max_len]
        else:
            padded_seqs[i, :len(seq)] = seq
    return padded_seqs

def data_process(src_texts, trg_texts, src_vocab, trg_vocab, src_language='english', trg_language='german', max_len=100, pad_idx=0) -> tuple:
    '''
    Args:
        src_texts: list of str
        trg_texts: list of str
        src_vocab: dict, {token: idx}
        trg_vocab: dict, {token: idx}
    Returns:
        src_idx_list: list of longtensor
        trg_idx_list: list of longtensor
    '''
    
    
    src_tokens_list = [['<sos>'] + tokenize(text, language=src_language) + ['<eos>'] for text in src_texts]
    trg_tokens_list = [['<sos>'] + tokenize(text, language=trg_language) + ['<eos>'] for text in trg_texts]
    src_idx_list = [tokens2idx(tokens, src_vocab) for tokens in src_tokens_list]
    trg_idx_list = [tokens2idx(tokens, trg_vocab) for tokens in trg_tokens_list]
    
    return src_idx_list, trg_idx_list

def generate_vocab(texts, language='english', max_vocab_size=10000) -> dict:
    '''
    Args:
        texts: list of str
    Returns:
        dict, {token: idx}
    '''
    tokens = [token for text in texts for token in tokenize(text, language=language)]
    freq = nltk.FreqDist(tokens)
    vocab = {token: idx for idx, (token, _) in enumerate(freq.most_common(max_vocab_size-4), 4)}
    vocab['<pad>'] = 0
    vocab['<sos>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3
    return vocab

def generate_vocab_from_file(file_path, language='english', max_vocab_size=10000) -> dict:
    with open(file_path, 'r') as f:
        texts = f.readlines()
    return generate_vocab(texts, language=language, max_vocab_size=max_vocab_size)

def generate_vocab_from_json_files(directory, language='english', max_vocab_size=10000) -> dict:
    count = {}
    for file_name in os.listdir(directory):
        print(f'processing {file_name}...')
        with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [pair[language] for pair in data]
        tokens = [token for text in texts for token in tokenize(text, language=language)]
        for token in tokens:
            if token in count:
                count[token] += 1
            else:
                count[token] = 1
        print(f'Done. {file_name} has {len(tokens)} tokens.')
    # based on count, choose the most frequent tokens
    count = dict(sorted(count.items(), key=lambda x: x[1], reverse=True))
    vocab = {token: idx for idx, token in enumerate(list(count.keys())[: max_vocab_size-4], 4)}
    print(f'Done. Vocab size: {len(vocab)}')
    vocab['<pad>'] = 0
    vocab['<sos>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3
    return vocab

# test the generate_vocab_from_json_files
if __name__ == '__main__':
    directory = r'../data/translation2019zh_train_split'
    vocab = generate_vocab_from_json_files(directory, language='english', max_vocab_size=10000)
    breakpoint()    
        
    
    
    
    
    
def load_vocab(file_path) -> dict:
    with open(file_path, 'r') as f:
        vocab = {line.strip(): idx for idx, line in enumerate(f)}
    return vocab

def save_vocab(vocab, file_path):
    with open(file_path, 'w') as f:
        for token, idx in vocab.items():
            f.write(f'{token}\n')

def json_file_split(file_path, target_directory, each_size=100000):
    '''
    Args:
        file_path: str, path of the json file
        target_directory: str, path of the target directory
        each_size: int, the size of each file
    
    Description: split a huge json file into multiple files
    
    '''
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    
    chunk = []
    file_count = 1
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                chunk.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
            
            if len(chunk) >= each_size:
                output_file_path = os.path.join(target_directory, f'{base_name}_part_{file_count}{ext}')
                with open(output_file_path, 'w', encoding='utf-8') as out_f:
                    json.dump(chunk, out_f, ensure_ascii=False, indent=4)
                print(f'Saved {output_file_path}')
                chunk = []
                file_count += 1
    
    # Save any remaining data
    if chunk:
        output_file_path = os.path.join(target_directory, f'{base_name}_part_{file_count}{ext}')
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            json.dump(chunk, out_f, ensure_ascii=False, indent=4)
        print(f'Saved {output_file_path}')


    