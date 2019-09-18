# -*- coding:utf-8 -*-

import numpy as np
import logging
import json
from mltools import Parameter
from marginal_functions import get_bits

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
        self.layers_matrix_sizes = self.params.layers_matrix_sizes

        from expectations import data_expectations as de
        from expectations import model_expectations as me
        # そのうち引数で変えられるようにすべき
        self.data_expectation = de.mean_field
        self.model_expectation = me.montecarlo

        self._probs = None
        self._old_probs_id = None

    def train(self, data, train_time, optimizer, minibatch_size=100):
        for i in range(train_time):
            data_exp = self.data_expectation(self, data.minibatch(minibatch_size))
            model_exp = self.model_expectation(self)
            diff = optimizer.update( data_exp - model_exp )
            self.params += diff
            logging.info("learning time: %d"%i)
    
    # !!! exponential runnning time !!!
    # returns *all* patterns of P(v, h1, h2)
    def probability(self):
        if len(self.layers) != 3:
            raise TypeError("exact method only supports 3-layer DBM.")

        if id(self.params.weights) == self._old_probs_id:
            logging.debug("probability calculation skipped.")
            return self._probs
        
        energies = [None for i in range(len(self.params.weights))]
        bits = get_bits(np.max(self.layers))

        for i in range(len(self.layers)-1):
            upper = self.layers[i+1]
            lower = self.layers[i]
            energies[i] = np.dot( np.dot(bits[0:2**lower, 0:lower], self.params.weights[i]), bits[0:2**upper, 0:upper].T )
        
        energy = energies[0][:, :, np.newaxis] + energies[1][np.newaxis, :, :]
        energy_exp = np.exp(energy - np.max(energy))
        probability = energy_exp / np.sum(energy_exp)
        
        self._probs = probability
        self._old_probs_id = id(self.params.weights)

        return probability
    
    # !!! exponential runnning time !!!
    def kl_divergence(self, gen_dbm):
        probs = np.sum(self.probability(), axis=(1,2))
        gen_probs = np.sum(gen_dbm.probability(), axis=(1,2))
        return np.sum( gen_probs * np.log( gen_probs / probs ) )
    
    def save(self, filename):
        params_listed = {}
        for w in self.params.params:
            params_listed[w] = self.params.params[w].tolist()
        data = {
            "layers": self.layers.tolist(),
            "params": params_listed
        }
        json.dump(data, open(filename, "w+"), indent=2)
    
    @staticmethod
    def load(filename):
        data = json.load(open(filename, "r"))
        for w in data["params"]:
            data["params"][w] = np.array(data["params"][w])
        return DBM(data["layers"], initial_params=data["params"])