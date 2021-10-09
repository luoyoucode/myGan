import torch
import torch.nn as nn

a = torch.ones(1,1,3,3)
print(a)
b = nn.Upsample(scale_factor=2,mode='nearest')
print(b(a))
