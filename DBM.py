# -*- coding:utf-8 -*-

import numpy as np
import logging
import json
from mltools import Parameter
from mltools import EpochCalc
from marginal_functions import get_bits

from numpy.lib.stride_tricks import as_strided

class DBM_params(Parameter):
    def __init__(self, layers, initial_params=None, gauss=False):
        self.layers = np.array(layers)
        # [1, 2, 3, 4] -> [[1, 2], [2, 3], [3, 4]]
        self.layers_matrix_sizes = as_strided(self.layers, (len(self.layers)-1, 2), (self.layers.strides[0], self.layers.strides[0]))
        
        if initial_params is None:
            self.weights = []
            for i, j in self.layers_matrix_sizes:
                if gauss:
                    sigma = np.sqrt( 1.0/(i+j) )
                    self.weights.append( np.random.normal(0, sigma, (i,j)) )
                else:
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
    def __init__(self, layers, data_expectation="mean_field", model_expectation="montecarlo", initial_params=None):
        self.params = DBM_params(layers, initial_params)
        self.layers = self.params.layers
        self.layers_matrix_sizes = self.params.layers_matrix_sizes

        from expectations import data_expectations as de
        from expectations import model_expectations as me
        self.data_expectation = getattr(de, data_expectation)
        self.model_expectation = getattr(me, model_expectation)

    def train(self, data, train_epoch, gen_dbm, learning_result, optimizer, minibatch_size=100, test_interval=1.0):
        ec = EpochCalc(train_epoch, len(data), minibatch_size)

        def per_epoch(update_time):
            logging.debug("Calculating Kullback-Leibler Divergence.")
            kld = self.kl_divergence(gen_dbm)
            learning_result.make_log(ec.update_to_epoch(update_time, force_integer=False), "kl-divergence", [kld])
            logging.debug("Calculating Log-Likelihood function.")
            ll = self.log_likelihood(data.data)
            learning_result.make_log(ec.update_to_epoch(update_time, force_integer=False), "log-likelihood", [ll])
            logging.info("[ {} / {} ]( {} / {} ) KL-Divergence: {}, log-likelihood: {}".format(ec.update_to_epoch(update_time, force_integer=False), ec.train_epoch, update_time, ec.train_update , kld, ll))

        per_epoch(0)
        for i in range(1, ec.train_update+1):
            logging.debug("training process : [ %d / %d ]" % (i, ec.train_update))
            _, minibatch_idx = data.minibatch(minibatch_size, return_idx=True)
            data_exp = self.data_expectation(self, data.data, minibatch_idx)
            model_exp = self.model_expectation(self)
            diff = optimizer.update( data_exp - model_exp )
            self.params += diff
            if i % ec.epoch_to_update(test_interval) == 0:
                per_epoch(i)
    
    # !!! exponential runnning time !!!
    # returns *all* patterns of P(v, h1, h2)
    def probability(self, data=None, data_is_conditional=False):
        if len(self.layers) != 3:
            raise TypeError("probability method only supports 3-layer DBM.")

        energies = [None for i in range(len(self.params.weights))]
        bits = get_bits(np.max(self.layers))

        for i in range(len(self.layers)-1):
            upper = self.layers[i+1]
            lower = self.layers[i]
            energies[i] = np.dot( np.dot(bits[0:2**lower, 0:lower], self.params.weights[i]), bits[0:2**upper, 0:upper].T )
        
        energy = energies[0][:, :, np.newaxis] + energies[1][np.newaxis, :, :]
        energy_max = np.max(energy)
        energy_exp = np.exp(energy - energy_max)
        state_sum = np.sum(energy_exp)

        if not data is None: 
            energies[0] = np.dot( np.dot(data, self.params.weights[0]), bits[0:2**self.layers[1], 0:self.layers[1]].T )
            energy = energies[0][:, :, np.newaxis] + energies[1][np.newaxis, :, :]
            if data_is_conditional:
                # normalization without visible layer.
                energy_exp = np.exp(energy - np.max(energy, axis=(1,2), keepdims=True))
                probability = energy_exp / np.sum(energy_exp, axis=(1,2), keepdims=True)
            else:
                energy_exp = np.exp(energy - energy_max)
                probability = energy_exp / state_sum

        else:
            probability = energy_exp / state_sum
        
        return probability
    
    # !!! exponential running time !!!
    def log_likelihood(self, data):
        # ln P(v)
        logprobs = np.log( np.sum(self.probability(data=data), axis=(1,2))  )
        return np.mean(logprobs)

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
    def load(filename, data_expectation="mean_field", model_expectation="montecarlo"):
        data = json.load(open(filename, "r"))
        for w in data["params"]:
            data["params"][w] = np.array(data["params"][w])
        return DBM(data["layers"], data_expectation, model_expectation, initial_params=data["params"])