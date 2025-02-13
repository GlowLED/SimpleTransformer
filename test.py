import torch
import torch.nn as nn
import nltk

a = '你是谁？'
b = list(a)
print(b)
c = 'Hello, World!'
d = nltk.word_tokenize(c, language='english')
print(d)