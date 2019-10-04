# -*- coding:utf-8 -*-

import numpy as np
from DBM import DBM_params
from marginal_functions import sigmoid, act, get_bits

class data_expectations:
    old_means = None
    old_samples = None

    @classmethod
    def _sampling(cls, dbm, data, update_time):
        sample_num = len(data)
        if cls.old_samples is None:
            initial_vectors = [np.random.choice([-1, 1], i) for i in dbm.layers[1:]]
            values = [np.tile(i, (sample_num, 1)) for i in initial_vectors]
        else:
            values = cls.old_samples
            update_time = 1

        for i in range(update_time):

            act_prob = act( np.dot(data, dbm.params.weights[0]) + np.dot(dbm.params.weights[1], values[1].T).T )
            values[0] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[1]), 1, -1)

            for l in range(1, len(dbm.layers)-2):
                act_prob = act( np.dot( values[l-1], dbm.params.weights[l] ) + np.dot(dbm.params.weights[l+1], values[l+1].T).T )
                values[l] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[l+1]), 1, -1)

            act_prob = act( np.dot( values[-2], dbm.params.weights[-1] ) )
            values[-1] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[-1]), 1, -1)

        cls.old_samples = values
        return values

    @classmethod
    def mean_field(cls, dbm, data, approximition_time=100):
        data_length = len(data)
        if cls.old_means is None:
            means = [np.random.randn(data.shape[0],r) for r in dbm.layers[1:]]
        else:
            means = cls.old_means
        expectations = [None for i in range(len(dbm.params.weights))]
        for t in range(approximition_time):
            # (N, i)(i, j) + (N, k)(j, k)^T
            means[0] = np.tanh( np.dot(data, dbm.params.weights[0]) + np.dot(means[1], dbm.params.weights[1].T) )
            for i in range(1, len(dbm.layers)-2):
                means[i] = np.tanh( np.dot(means[i-1], dbm.params.weights[i] ) + np.dot(means[i+1], dbm.params.weights[i+1].T) )
            means[-1] = np.tanh( np.dot(means[-2], dbm.params.weights[-1]) )
        cls.old_means = means

        for e in range(len(expectations)):
            if e==0:
                expectations[e] = np.dot(data.T, means[e]) / data_length
            else:
                expectations[e] = np.dot(means[e-1].T, means[e]) / data_length
        return DBM_params(dbm.layers, initial_params=(expectations))

    @classmethod
    def smci(cls, dbm, data):
        values = cls._sampling(dbm, data, update_time=1000)
        
        means = [None for r in dbm.layers[1:]]
        means[0] = np.tanh( np.dot(data, dbm.params.weights[0]) + np.dot(values[1], dbm.params.weights[1].T) )
        for i in range(1, len(dbm.layers)-2):
            means[i] = np.tanh( np.dot(values[i-1], dbm.params.weights[i-1] ) + np.dot(values[i+1], dbm.params.weights[i+1].T) )
        means[-1] = np.tanh( np.dot(values[-2], dbm.params.weights[-1]) )

        expectations = [None for i in dbm.params.weights]
        for e,_ in enumerate(expectations):
            if e==0:
                expectations[e] = np.dot(data.T, means[e]) / len(data)
            else:
                expectations[e] = np.dot(means[e-1].T, means[e]) / len(data)
        return DBM_params(dbm.layers, initial_params=(expectations))

    # !!! this function takes exponential running time and requires huge memory !!!
    @classmethod
    def exact(cls, dbm, data):
        if len(dbm.layers) != 3:
            raise TypeError("exact method only supports 3-layer DBM.")

        expectations = [np.zeros((i,j)) for i,j in dbm.layers_matrix_sizes]
        bits = get_bits(np.max(dbm.layers))

        probability = dbm.probability(data)

        lbits = [bits[0:2**dbm.layers[i], 0:dbm.layers[i]] for i in range(1, len(dbm.layers))]

        for d in range(len(data)):
            for h1 in range(2**dbm.layers[1]):
                for h2 in range(2**dbm.layers[2]):
                    np.add(expectations[0], np.outer(data[d], lbits[0][h1]) * probability[d][h1][h2], out=expectations[0])
                    np.add(expectations[1], np.outer(lbits[0][h1], lbits[1][h2]) * probability[d][h1][h2], out=expectations[1])

        for e,_ in enumerate(expectations):
            expectations[e] /= len(data)

        return DBM_params(dbm.layers, initial_params=(expectations))

class model_expectations:
    old_samples = None

    @classmethod
    def _sampling(cls, dbm, update_time, sample_num):
        if cls.old_samples is None:
            initial_vectors = [np.random.choice([-1, 1], i) for i in dbm.layers]
            values = [np.tile(i, (sample_num, 1)) for i in initial_vectors]
        else:
            values = cls.old_samples
            update_time = 1

        for i in range(update_time):

            act_prob = act( np.dot(dbm.params.weights[0], values[1].T).T )
            values[0] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[0]), 1, -1)

            for l in range(1, len(dbm.layers)-1):
                act_prob = act( np.dot( values[l-1], dbm.params.weights[l-1] ) + np.dot(dbm.params.weights[l], values[l+1].T).T )
                values[l] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[l]), 1, -1)

            act_prob = act( np.dot( values[-2], dbm.params.weights[-1] ) )
            values[-1] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[-1]), 1, -1)

        cls.old_samples = values
        return values

    @classmethod
    def montecarlo(cls, dbm, update_time=1000, sample_num=1000):
        values = cls._sampling(dbm, update_time, sample_num)
        expectations = [None for i in range(len(dbm.params.weights))]
        for e,_ in enumerate(expectations):
            expectations[e] = np.dot(values[e].T, values[e+1]) / sample_num
        return DBM_params(dbm.layers, initial_params=(expectations))
    
    @classmethod
    def smci(cls, dbm, update_time=1000, sample_num=1000):
        values = cls._sampling(dbm, update_time, sample_num)
        
        means = [None for r in dbm.layers]
        means[0] = np.tanh( np.dot(values[1], dbm.params.weights[0].T) )
        for i in range(1, len(dbm.layers)-1):
            means[i] = np.tanh( np.dot(values[i-1], dbm.params.weights[i-1] ) + np.dot(values[i+1], dbm.params.weights[i].T) )
        means[-1] = np.tanh( np.dot(values[-2], dbm.params.weights[-1]) )

        expectations = [None for i in dbm.params.weights]
        for e,_ in enumerate(expectations):
            expectations[e] = np.dot(means[e].T, means[e+1]) / sample_num
        return DBM_params(dbm.layers, initial_params=(expectations))

    # !!! this function takes exponential running time and requires HUGE memory !!!
    @classmethod
    def exact(cls, dbm):
        if len(dbm.layers) != 3:
            raise TypeError("exact method only supports 3-layer DBM.")

        expectations = [np.zeros((i,j)) for i,j in dbm.layers_matrix_sizes]
        bits = get_bits(np.max(dbm.layers))
        probability = dbm.probability()

        lbits = [bits[0:2**dbm.layers[i], 0:dbm.layers[i]] for i in range(len(dbm.layers))]
        for v in range(2**dbm.layers[0]):
            for h1 in range(2**dbm.layers[1]):
                for h2 in range(2**dbm.layers[2]):
                    np.add(expectations[0], np.outer(lbits[0][v], lbits[1][h1]) * probability[v][h1][h2], out=expectations[0])
                    np.add(expectations[1], np.outer(lbits[1][h1], lbits[2][h2]) * probability[v][h1][h2], out=expectations[1])

        return DBM_params(dbm.layers, initial_params=(expectations))