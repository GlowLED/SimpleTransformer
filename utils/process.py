import torch
import nltk
import zhconv
def padding(seq_list, pad_idx=0):
    '''
    Args:
        seq_list: list of list of int, each int is a token idx
        pad_idx: int
    
    Returns:
        padded_seq: tensor [batch_size, max_len]
        max_len: int
    '''
    max_len = max([len(seq) for seq in seq_list])
    
    padded_seq = torch.ones(len(seq_list), max_len, dtype=torch.long) * pad_idx
    
    for i, seq in enumerate(seq_list):
        padded_seq[i, :len(seq)] = torch.tensor(seq)
    
    return padded_seq, max_len


def tokenize(text, language='english'):
    if isinstance(text, str):
        if language == 'english':
            return nltk.word_tokenize(text, language='english')
        elif language == 'chinese':
            return list(text)
        else:
            raise ValueError('language must be english or chinese')
    elif isinstance(text, list):
        if language == 'english':
            return [nltk.word_tokenize(sent, language='english') for sent in text]
        elif language == 'chinese':
            return [list(sent) for sent in text]
        else:
            raise ValueError('language must be english or chinese')
    else:
        raise ValueError('text must be str or list')

def tokens2idx(tokens: list, vocab: dict):
    idx = [vocab.get(token, vocab['<unk>']) for token in tokens]
    return idx

def prepare_for_train(
    src_texts: list, trg_texts: list, src_vocab: dict, trg_vocab: dict,
    src_language='english', trg_language='chinese',
    bos_idx=1, eos_idx=2, pad_idx=0
):
    src_tokens = tokenize(src_texts, src_language)
    trg_tokens = tokenize(trg_texts, trg_language)
    
    src_idx = [tokens2idx(tokens, src_vocab) for tokens in src_tokens]
    trg_idx = [tokens2idx(tokens, trg_vocab) for tokens in trg_tokens]
    
    src_idx = [[bos_idx] + idx + [eos_idx] for idx in src_idx]
    trg_idx_input = [[bos_idx] + idx for idx in trg_idx]
    trg_idx_target = [idx + [eos_idx] for idx in trg_idx]
    
    src_idx, _ = padding(src_idx, pad_idx)
    trg_idx_input, _ = padding(trg_idx_input, pad_idx)
    trg_idx_target, _ = padding(trg_idx_target, pad_idx)
    
    return src_idx, trg_idx_input, trg_idx_target


def generate_vocab_from_dataset(dataset, src_language='english', trg_language='chinese', max_vocab_size=10000):
    src_vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    trg_vocab = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    src_count = {}
    trg_count = {}
    for i in range(len(dataset)):
        src_text, trg_text = dataset[i]
        src_tokens = tokenize(src_text, src_language)
        trg_tokens = tokenize(trg_text, trg_language)
        for token in src_tokens:
            if token not in src_count:
                src_count[token] = 1
            else:
                src_count[token] += 1
        for token in trg_tokens:
            if token not in trg_count:
                trg_count[token] = 1
            else:
                trg_count[token] += 1

    for token in list(src_count.keys()):
        if src_count[token] < 2:
            del src_count[token] 
    for token in list(trg_count.keys()):
        if trg_count[token] < 2:
            del trg_count[token]
    
    for token in sorted(src_count, key=src_count.get, reverse=True):
        if len(src_vocab) == max_vocab_size:
            break
        if token not in src_vocab:
            src_vocab[token] = len(src_vocab)
    
    for token in sorted(trg_count, key=trg_count.get, reverse=True):
        if len(trg_vocab) == max_vocab_size:
            break
        if token not in trg_vocab:
            trg_vocab[token] = len(trg_vocab)
    
    return src_vocab, trg_vocab

def idx2tokens(idx: list, de_vocab: dict):
    tokens = [de_vocab.get(idx_) for idx_ in idx]
    return tokens

def detokenize(tokens: list, language):
    if language == 'english':
        return ' '.join(tokens)
    elif language == 'chinese':
        return ''.join(tokens)
    else:
        raise ValueError('language must be english or chinese')

def traditional_to_simplified(traditional_text):
    simplified_text = zhconv.convert(traditional_text, 'zh-hans')
    return simplified_text
