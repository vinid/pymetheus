import numpy as np
import torch
from torch import nn

from pymetheus.logics.interfaces import Logic
from pymetheus.utils.functionalities import get_torch_device


class LukasiewiczLogic(Logic):
    """
    Implements the Fuzzy logic defined by Lukasiewicz.
    """
    class Negation(nn.Module):
        def __init__(self):
            super(LukasiewiczLogic.Negation, self).__init__()

        def forward(self, x):
            return 1 - x

    class TNorm(nn.Module):
        def __init__(self):
            super(LukasiewiczLogic.TNorm, self).__init__()

        def forward(self, x, y):
            assert x.shape == y.shape
            baseline = torch.from_numpy(np.array([0])).type(torch.FloatTensor).to(get_torch_device())

            val = x + y - 1

            return torch.max(baseline, val)

    class TConorm(nn.Module):
        def __init__(self):
            super(LukasiewiczLogic.TConorm, self).__init__()

        def forward(self, x, y):
            assert x.shape == y.shape
            baseline = torch.from_numpy(np.array([1])).type(torch.FloatTensor).to(get_torch_device())
            return torch.max(baseline, x + y)

    class Equal(nn.Module):
        def __init__(self):
            super(LukasiewiczLogic.Equal, self).__init__()

        def forward(self, x, y):
            assert x.shape == y.shape
            return (1 - torch.abs(x - y))

    class Residual(nn.Module):
        # Luk
        def __init__(self):
            super(LukasiewiczLogic.Residual, self).__init__()

        def forward(self, x, y):
            assert x.shape == y.shape
            baseline = torch.from_numpy(np.array([1])).type(torch.FloatTensor).to(get_torch_device())
            return torch.min(baseline, 1 - x + y)
