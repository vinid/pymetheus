from torch import nn
import numpy as np
import torch

class Negation(nn.Module):
    def __init__(self):
        super(Negation, self).__init__()

    def forward(self, x):
        return 1 - x


class T_Norm(nn.Module):
    # Luk
    def __init__(self):
        super(T_Norm, self).__init__()

    def forward(self, x, y):
        assert x.shape == y.shape
        baseline = torch.from_numpy(np.array([0])).type(torch.FloatTensor)

        val = x + y - 1

        return torch.max(baseline, val)

class T_CoNorm(nn.Module):
    # Luk
    def __init__(self):
        super(T_CoNorm, self).__init__()

    def forward(self, x, y):
        assert x.shape == y.shape
        baseline = torch.from_numpy(np.array([1])).type(torch.FloatTensor)
        return torch.max(baseline, x + y)

class Equal(nn.Module):
    def __init__(self):
        super(Equal, self).__init__()

    def forward(self, x, y):
        assert x.shape == y.shape

        return (1 - torch.abs(x - y))

class Residual(nn.Module):
    # Luk
    def __init__(self):
        super(Residual, self).__init__()

    def forward(self, x, y):
        baseline = torch.from_numpy(np.array([1])).type(torch.FloatTensor)
        assert x.shape == y.shape
        return torch.min(baseline, 1 - x + y)
