import torch
import numpy as np

t = torch.tensor(3)

t = torch.rand(1)

print (t)

print (t.item())

t1 = torch.rand(1, 1, 1, 1)

dim1 = torch.rand(1, 1, 1, 1).dim()

print(t1)

print(dim1)

t2 = torch.rand(6)

print(t2)

t3 = torch.rand(2, 3, 4, 5)

print(t3)


print(torch.rand(2, 3, 4, 5).dim())

print(torch.rand(2, 3, 4, 5).numel())

t4 = torch.ones(2, 4)

print(t4)

t5 = torch.rand(2, 3)

t_np = t5.numpy()

print (t5)

print (t_np)

