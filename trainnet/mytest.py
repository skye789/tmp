import numpy as np
import matplotlib.pyplot as plt
from phantominator import shepp_logan

from pygrappa import grappa

if __name__ == '__main__':
    nc = 20
    for i in range(nc):
        row = i//5+1
        col = i%5+1
        print(row, col)