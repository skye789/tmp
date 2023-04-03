"""
Python 3.x uses "round half to even", also called "Banker's rounding",
while MatLab uses "round away from zero".
Fallback for unknown/incompatible types is Python's inbuilt round
function.
"""

import importlib.util
import sys
from math import floor, ceil

import numpy as np

if importlib.util.find_spec('torch') is not None:
    import torch


def away_from_zero_torch(x):
    if isinstance(x, torch.Tensor):
        mask = x >= 0
        y = torch.zeros_like(x)
        y[mask] = torch.floor(x[mask] + 0.5)
        y[~mask] = torch.ceil(x[~mask] - 0.5)
        return y
    else:
        # fall back to numpy version
        return away_from_zero_numpy(x)


def away_from_zero_numpy(x):
    if isinstance(x, np.ndarray):
        mask = x >= 0
        y = np.zeros_like(x)
        y[mask] = np.floor(x[mask] + 0.5)
        y[~mask] = np.ceil(x[~mask] - 0.5)
        return y
    else:
        # fall back to python version
        return away_from_zero_python(x)


def away_from_zero_python(x):
    try:
        if x >= 0.0:
            return floor(x + 0.5)
        else:
            return ceil(x - 0.5)
    except TypeError:
        print('WARNING: Could not round away from zero. Using inbuilt banker\'s rounding instead.',
              file=sys.stderr, flush=True)
        return round(x)
