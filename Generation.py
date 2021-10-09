import numpy.random
import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((-1,)+self.shape)


class Generation(nn.Module):
    def __init__(self):
        super(Generation, self).__init__()
        self.Dense = nn.Sequential(
            nn.Linear(100, 128 * 16 * 16),
            Reshape(128, 16, 16),
            nn.ReLU(),
        )

        self.GenStruct = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.Dense(x)
        x = torch.Tensor(x)
        x = self.GenStruct(x)
        return x

model = Generation()

a = torch.ones(64,100)
b = model(a)
print(b.size())
