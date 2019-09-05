# -*- coding:utf-8 -*-

import numpy as np
from DBM import DBM_params

def sigmoid(x):
    return 1/(1+np.exp(-x))

def act(x):
    return ( np.tanh(x) + 1 ) / 2

def get_bits(num_bits):
    x = np.arange(2**num_bits).reshape(-1, 1)
    to_and = 2**np.arange(num_bits).reshape(1, num_bits)
    bit_bools = (x & to_and).astype(bool)
    return np.where(bit_bools, 1, -1)

class data_expectations:
    @staticmethod
    def mean_field(dbm, data, approximition_time=1000):
        data_length = len(data)
        means = [np.random.randn(data.shape[0],r) for r in dbm.layers[1:]]
        expectations = [None for i in range(len(dbm.weights))]
        for t in range(approximition_time):
            # (N, i)(i, j) + (N, k)(j, k)^T
            means[0] = np.tanh( np.dot(data, dbm.weights[0]) + np.dot(means[1], dbm.weights[1].T) )
            for i in range(1, len(dbm.layers)-2):
                means[i] = np.tanh( np.dot(means[i-1], dbm.weights[i] ) + np.dot(means[i+1], dbm.weights[i+1].T) )
            means[-1] = np.tanh( np.dot(means[-2], dbm.weights[-1]) )

        for e in range(len(expectations)):
            if e==0:
                expectations[e] = np.dot(data.T, means[e]) / data_length
            else:
                expectations[e] = np.dot(means[e-1].T, means[e]) / data_length
        return DBM_params(dbm.layers, initial_params=(expectations))

    # !!! this function takes exponential running time and requires huge memory !!!
    @staticmethod
    def exact(dbm, data):
        if len(dbm.layers) != 3:
            raise TypeError("exact method only supports 3-layer DBM.")

        expectations = [np.zeros((i,j)) for i,j in dbm.layers_matrix_sizes]
        energies = [None for i in range(len(dbm.weights))]
        bits = get_bits(np.max(dbm.layers))

        energies[0] = np.dot( np.dot(data, dbm.weights[0]), bits[0:2**dbm.layers[1], 0:dbm.layers[1]].T )
        for i in range(1, len(dbm.layers)-1):
            upper = dbm.layers[i+1]
            lower = dbm.layers[i]
            energies[i] = np.dot( np.dot(bits[0:2**lower, 0:lower], dbm.weights[i]), bits[0:2**upper, 0:upper].T )
        
        energy = energies[0][:, :, np.newaxis] + energies[1][np.newaxis, :, :]
        energy_exp = np.exp(energy - np.max(energy, axis=(1,2), keepdims=True))
        probability = energy_exp / np.sum(energy_exp, axis=(1,2), keepdims=True)

        lbits = [bits[0:2**dbm.layers[i], 0:dbm.layers[i]] for i in range(1, len(dbm.layers))]

        for d in range(len(data)):
            for h1 in range(2**dbm.layers[1]):
                for h2 in range(2**dbm.layers[2]):
                    np.add(expectations[0], np.outer(data[d], lbits[0][h1]) * probability[d][h1][h2], out=expectations[0])
                    np.add(expectations[1], np.outer(lbits[0][h1], lbits[1][h2]) * probability[d][h1][h2], out=expectations[1])

        for e,_ in enumerate(expectations):
            expectations[e] /= len(data)

        return expectations

class model_expectations:
    @staticmethod
    def montecarlo(dbm, update_time=1000, sample_num=1000, initial_values=None):
        if initial_values is None:
            initial_vectors = [np.random.choice([-1, 1], i) for i in dbm.layers]
            values = [np.tile(i, (sample_num, 1)) for i in initial_vectors]
        else:
            values = initial_values
            update_time = 1

        for i in range(update_time):

            act_prob = act( np.dot(dbm.weights[0], values[1].T).T )
            values[0] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[0]), 1, -1)

            for l in range(1, len(dbm.layers)-1):
                act_prob = act( np.dot( values[l-1], dbm.weights[l-1] ) + np.dot(dbm.weights[l], values[l+1].T).T )
                values[l] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[l]), 1, -1)

            act_prob = act( np.dot( values[-2], dbm.weights[-1] ) )
            values[-1] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[-1]), 1, -1)

        expectations = [None for i in range(len(dbm.weights))]
        for e,_ in enumerate(expectations):
            expectations[e] = np.dot(values[e].T, values[e+1]) / sample_num
        return DBM_params(dbm.layers, initial_params=(expectations)), values

    # !!! this function takes exponential running time and requires HUGE memory !!!
    @staticmethod
    def exact(dbm):
        if len(dbm.layers) != 3:
            raise TypeError("exact method only supports 3-layer DBM.")
        
        expectations = [np.zeros((i,j)) for i,j in dbm.layers_matrix_sizes]
        energies = [None for i in range(len(dbm.weights))]
        bits = get_bits(np.max(dbm.layers))

        for i in range(len(dbm.layers)-1):
            upper = dbm.layers[i+1]
            lower = dbm.layers[i]
            energies[i] = np.dot( np.dot(bits[0:2**lower, 0:lower], dbm.weights[i]), bits[0:2**upper, 0:upper].T )
        
        energy = energies[0][:, :, np.newaxis] + energies[1][np.newaxis, :, :]
        energy_exp = np.exp(energy - np.max(energy))
        probability = energy_exp / np.sum(energy_exp)

        lbits = [bits[0:2**dbm.layers[i], 0:dbm.layers[i]] for i in range(len(dbm.layers))]
        for v in range(2**dbm.layers[0]):
            for h1 in range(2**dbm.layers[1]):
                for h2 in range(2**dbm.layers[2]):
                    np.add(expectations[0], np.outer(lbits[0][v], lbits[1][h1]) * probability[v][h1][h2], out=expectations[0])
                    np.add(expectations[1], np.outer(lbits[1][h1], lbits[2][h2]) * probability[v][h1][h2], out=expectations[1])

        return expectations