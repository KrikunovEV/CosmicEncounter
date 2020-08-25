import torch
import numpy as np

data = np.arange(15) - 5
np.random.shuffle(data)
a = torch.Tensor(data)

print(a)
print(torch.tanh(a))
print(torch.tanh(a).view(3, 5))
print(torch.tanh(a).view(3, 5).prod(0))