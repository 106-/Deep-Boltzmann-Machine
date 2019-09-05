#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from expectations import data_expectations as de
from expectations import model_expectations as me

from numpy.lib.stride_tricks import as_strided

class DBM_params:
    def __init__(self, layers, initial_params=None):
        self.layers = np.array(layers)
        # [1, 2, 3, 4] -> [[1, 2], [2, 3], [3, 4]]
        self.layers_matrix_sizes = as_strided(self.layers, (len(self.layers)-1, 2), (self.layers.strides[0], self.layers.strides[0]))
        
        if initial_params is None:
            self.weights = []
            for i, j in self.layers_matrix_sizes:
                # xavier's initialization
                uniform_range = np.sqrt(6/(i+j))
                self.weights.append(np.random.uniform(-uniform_range, uniform_range, (i, j)))
        else:
            self.weights = initial_params["weights"]

        params = {}
        for i,w in enumerate(self.weights):
            params["weight%d"%i] = w
        
        super().__init__(params)
    
    def zeros(self):
        zero_params = {}
        for i in self.params:
            zero_params[i] = np.zeros(self.[i].shape)
        return DBM_params(self.layers, initial_params=zero_params)

class DBM:
    def __init__(self, layers, initial_params=None):
        self.params = DBM_params(layers, initial_params)
        self.layers = self.params.layers
        self.weights = self.params.weights
        self.layers_matrix_sizes = self.params.layers_matrix_sizes

    def data_expectation(self, data, method=de.mean_field, **kwargs):
        return method(self, data, **kwargs)

    def model_expectation(self, method=me.montecarlo, **kwargs):
        return method(self, **kwargs)

    def train(self, train_time):
        old_samples = None
        for i in range(train_time):
            pass

def main():
    import time
    dbm = DBM([5, 5, 5])
    data = np.random.choice([-1, +1], (100, 5))
    a = dbm.data_expectation(data, method=de.mean_field)
    b = dbm.data_expectation(data, method=de.exact)
    # a,_ = dbm.model_expectation(method=me.montecarlo)
    # b = dbm.model_expectation(method=me.exact)
    for i in range(len(dbm.weights)):
        print("="*20)
        print(a[i])
        print(b[i])
        print("="*20)

if __name__=="__main__":
    main()