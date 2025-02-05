'''
some functions to process text data
for example, tokenize, detokenize, generate_vocab, data_process

'''


import torch
import torch.nn.functional as F
import nltk
import jieba
import logging

jieba.setLogLevel(logging.INFO)


def tokenize(text, language='english'):
    if language == 'chinese':
        return list(jieba.cut(text))
    return nltk.word_tokenize(text, language=language)

def detokenize(tokens):
    return ' '.join(tokens)

def tokens2idx(tokens, en_vocab):
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

def idx2tokens(idx, de_vocab):
    '''
    Args:
        idx: longtensor
        de_vocab: dict, {idx: token}
    Returns:
        list of str
    '''
    tokens = [de_vocab[i.item()] for i in idx]
    return tokens

def padding(seqs, max_len, pad_idx=0):
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

def data_process(src_texts, trg_texts, src_vocab, trg_vocab, src_language='english', trg_language='german', max_len=100, pad_idx=0):
    '''
    Args:
        src_texts: list of str
        trg_texts: list of str
        src_vocab: dict, {token: idx}
        trg_vocab: dict, {token: idx}
    Returns:
        src_padded: longtensor of [len(src_texts), max_len]
        trg_padded: longtensor of [len(trg_texts), max_len]
    '''
    
    
    src_tokens_list = [['<sos>'] + tokenize(text, language=src_language) + ['<eos>'] for text in src_texts]
    trg_tokens_list = [['<sos>'] + tokenize(text, language=trg_language) + ['<eos>'] for text in trg_texts]
    src_idx_list = [tokens2idx(tokens, src_vocab) for tokens in src_tokens_list]
    trg_idx_list = [tokens2idx(tokens, trg_vocab) for tokens in trg_tokens_list]
    src_padded = padding(src_idx_list, max_len, pad_idx)
    trg_padded = padding(trg_idx_list, max_len, pad_idx)
    
    return src_padded, trg_padded

def generate_vocab(texts, language='english', max_vocab_size=10000):
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

def generate_vocab_from_file(file_path, language='english', max_vocab_size=10000):
    with open(file_path, 'r') as f:
        texts = f.readlines()
    return generate_vocab(texts, language=language, max_vocab_size=max_vocab_size)

def load_vocab(file_path):
    with open(file_path, 'r') as f:
        vocab = {line.strip(): idx for idx, line in enumerate(f)}
    return vocab

def save_vocab(vocab, file_path):
    with open(file_path, 'w') as f:
        for token, idx in vocab.items():
            f.write(f'{token}\n')
