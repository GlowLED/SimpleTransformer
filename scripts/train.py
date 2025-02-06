import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer.model import Transformer
from utils.process import tokenize, detokenize, tokens2idx, idx2tokens, padding

