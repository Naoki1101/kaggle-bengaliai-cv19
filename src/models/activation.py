import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU(Module):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)


class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x
