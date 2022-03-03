import torch
import numpy as np

print([] + [50, 100] + [1])

a = torch.rand(5, 4)
b = torch.zeros(1, 4, requires_grad=True)

print(a)

idx = 0
c = torch.cat([a[:idx], b, a[idx + 1:]], 0)
print(a[:idx].shape, b.shape, a[idx + 1:].shape)
print(c)