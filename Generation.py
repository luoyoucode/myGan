import numpy.random
import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Generation(nn.Module):
    def __init__(self):
        super(Generation, self).__init__()
        self.GenStruct = nn.Sequential(
            nn.Linear(100, 128 * 16 * 16),
            nn.ReLU(),
            Reshape(128, 16, 16),
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.Conv2d(128,128,kernel_size=3,padding=1,stride=1),
            nn.ReLU(),
            nn.MaxUnpool2d(kernel_size=2,stride=2),

        )

    def forward(self, inputs):
        output = self.GenStruct(inputs)
        return output


