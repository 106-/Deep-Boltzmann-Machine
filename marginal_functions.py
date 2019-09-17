# -*- coding:utf-8 -*-

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def act(x):
    return ( np.tanh(x) + 1 ) / 2

def get_bits(num_bits):
    x = np.arange(2**num_bits).reshape(-1, 1)
    to_and = 2**np.arange(num_bits).reshape(1, num_bits)
    bit_bools = (x & to_and).astype(bool)
    return np.where(bit_bools, 1, -1)