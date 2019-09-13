# -*- coding:utf-8 -*-

import numpy as np
from mltools import Parameter

from numpy.lib.stride_tricks import as_strided

class DBM_params(Parameter):
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
        elif isinstance(initial_params, dict):
            self.weights = [initial_params[i] for i in sorted(initial_params)]
        elif isinstance(initial_params, list):
            self.weights = initial_params
        else:
            raise TypeError("initial_params is unknown type: %s"%type(initial_params))

        digit = len(str(len(self.layers)))
        params = {}
        for i,w in enumerate(self.weights):
            params["weight%s"%str(i).zfill(digit)] = w
        
        super().__init__(params)
    
    def zeros(self):
        zero_params = {}
        for i in self.params:
            zero_params[i] = np.zeros(self[i].shape)
        return DBM_params(self.layers, initial_params=zero_params)

class DBM:
    def __init__(self, layers, initial_params=None):
        self.params = DBM_params(layers, initial_params)
        self.layers = self.params.layers
        self.weights = self.params.weights
        self.layers_matrix_sizes = self.params.layers_matrix_sizes

        from expectations import data_expectations as de
        from expectations import model_expectations as me
        # そのうち引数で変えられるようにすべき
        self.data_expectation = de.mean_field
        self.model_expectation = me.montecarlo

    def train(self, data, train_time, optimizer, minibatch_size=100):
        for i in range(train_time):
            data_exp = self.data_expectation(self, data.minibatch(minibatch_size))
            model_exp = self.model_expectation(self)
            diff = optimizer.update( data_exp - model_exp )
            self.params += diff

