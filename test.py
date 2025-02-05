import torch

a = torch.randn(size=(2, 4, 4))
mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
mask & mask
