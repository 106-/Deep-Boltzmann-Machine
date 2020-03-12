# -*- coding:utf-8 -*-

import numpy as np
from DBM import DBM_params
from marginal_functions import sigmoid, act, get_bits, tantan

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
            # 往路
            act_prob = act( np.dot(data, dbm.params.weights[0]) + np.dot(dbm.params.weights[1], values[1].T).T )
            values[0] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[1]), 1, -1)

            for l in range(1, len(dbm.layers)-2):
                act_prob = act( np.dot( values[l-1], dbm.params.weights[l] ) + np.dot(dbm.params.weights[l+1], values[l+1].T).T )
                values[l] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[l+1]), 1, -1)

            act_prob = act( np.dot( values[-2], dbm.params.weights[-1] ) )
            values[-1] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[-1]), 1, -1)

            # 復路
            for l in reversed(range(1, len(dbm.layers)-2)):
                act_prob = act( np.dot( values[l-1], dbm.params.weights[l] ) + np.dot(dbm.params.weights[l+1], values[l+1].T).T )
                values[l] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[l+1]), 1, -1)

            act_prob = act( np.dot(data, dbm.params.weights[0]) + np.dot(dbm.params.weights[1], values[1].T).T )
            values[0] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[1]), 1, -1)

        cls.old_samples = values
        return values

    @classmethod
    def mean_field(cls, dbm, data, data_idx, approximition_time=100):
        data_length = len(data_idx)
        if cls.old_means is None:
            means = [np.random.randn(data.shape[0],r) for r in dbm.layers[1:]]
        else:
            means = cls.old_means
        expectations = [None for i in dbm.params.weights]
        for t in range(approximition_time):
            # (N, i)(i, j) + (N, k)(j, k)^T
            means[0] = np.tanh( np.dot(data, dbm.params.weights[0]) + np.dot(means[1], dbm.params.weights[1].T) )
            for i in range(1, len(dbm.layers)-2):
                means[i] = np.tanh( np.dot(means[i-1], dbm.params.weights[i] ) + np.dot(means[i+1], dbm.params.weights[i+1].T) )
            means[-1] = np.tanh( np.dot(means[-2], dbm.params.weights[-1]) )

            for i in reversed(range(1, len(dbm.layers)-2)):
                means[i] = np.tanh( np.dot(means[i-1], dbm.params.weights[i] ) + np.dot(means[i+1], dbm.params.weights[i+1].T) )
            means[0] = np.tanh( np.dot(data, dbm.params.weights[0]) + np.dot(means[1], dbm.params.weights[1].T) )
        cls.old_means = means

        for e in range(len(expectations)):
            if e==0:
                expectations[e] = np.dot(data[data_idx].T, means[e][data_idx]) / data_length
            else:
                expectations[e] = np.dot(means[e-1][data_idx].T, means[e][data_idx]) / data_length
        return DBM_params(dbm.layers, initial_params=(expectations))

    @classmethod
    def smci(cls, dbm, data, data_idx):
        if len(dbm.layers) != 3:
            raise TypeError("smci method only supports 3-layer DBM.")
        values = cls._sampling(dbm, data, update_time=1000)

        mdata = data[data_idx]
        mvalues = [v[data_idx] for v in values]

        expectations = [None for i in dbm.params.weights]

        node_exp = np.tanh( np.dot(mdata, dbm.params.weights[0]) + np.dot( mvalues[1], dbm.params.weights[1].T) )
        upper = mdata[:, :, np.newaxis] * node_exp[:, np.newaxis, :]
        expectations[0] = np.mean(upper, axis=0)

        # dot(h1, w[1]) - h1 * w[1]
        upper = (np.dot(mvalues[0], dbm.params.weights[1])[:, np.newaxis, :] 
                    - mvalues[0][:, :, np.newaxis] * dbm.params.weights[1][np.newaxis, :, :])
        # dot(h2, w[1]) - h2 * w[1] + dot(data, w[0])
        under = (np.dot(mvalues[1], dbm.params.weights[1].T)[:, :, np.newaxis] 
                    - mvalues[1][:, np.newaxis, :] * dbm.params.weights[1][np.newaxis, :, :]
                    + np.dot(mdata, dbm.params.weights[0])[:, :, np.newaxis] )
        expectations[1] = np.mean(tantan(upper, under, dbm.params.weights[1]), axis=0)

        return DBM_params(dbm.layers, initial_params=(expectations))

    # !!! this function takes exponential running time and requires huge memory !!!
    @classmethod
    def exact(cls, dbm, data, data_idx):
        if len(dbm.layers) != 3:
            raise TypeError("exact method only supports 3-layer DBM.")
        mdata = data[data_idx]
        expectations = [None for i in dbm.params.weights]
        bits = get_bits(np.max(dbm.layers))
        probability = dbm.probability(mdata, True)

        # lbits: [(2**j, j), (2**k, k)]
        lbits = [bits[0:2**dbm.layers[i], 0:dbm.layers[i]] for i in range(1, len(dbm.layers))]

        # (N, 2**j, 2**k, i, j) -> (i, j)
        expectations[0] = np.sum(mdata[:, np.newaxis, np.newaxis, :, np.newaxis]
                            * lbits[0][np.newaxis, :, np.newaxis, np.newaxis, :]
                            * probability[:, :, :, np.newaxis, np.newaxis], axis=(0,1,2))
        # (N, 2**j, 2**k, j, k) -> (j, k)
        expectations[1] = np.sum(lbits[0][np.newaxis, :, np.newaxis, :, np.newaxis]
                            * lbits[1][np.newaxis, np.newaxis, :, np.newaxis, :]
                            * probability[:, :, :, np.newaxis, np.newaxis], axis=(0,1,2))

        for e,_ in enumerate(expectations):
            expectations[e] /= len(mdata)

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
            # 往路
            act_prob = act( np.dot(dbm.params.weights[0], values[1].T).T )
            values[0] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[0]), 1, -1)

            for l in range(1, len(dbm.layers)-1):
                act_prob = act( np.dot( values[l-1], dbm.params.weights[l-1] ) + np.dot(dbm.params.weights[l], values[l+1].T).T )
                values[l] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[l]), 1, -1)

            act_prob = act( np.dot( values[-2], dbm.params.weights[-1] ) )
            values[-1] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[-1]), 1, -1)

            # 復路
            for l in reversed(range(1, len(dbm.layers)-1)):
                act_prob = act( np.dot( values[l-1], dbm.params.weights[l-1] ) + np.dot(dbm.params.weights[l], values[l+1].T).T )
                values[l] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[l]), 1, -1)

            act_prob = act( np.dot(dbm.params.weights[0], values[1].T).T )
            values[0] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[0]), 1, -1)

        cls.old_samples = values
        return values

    @classmethod
    def montecarlo(cls, dbm, update_time=1000, sample_num=1000):
        values = cls._sampling(dbm, update_time, sample_num)
        expectations = [None for i in dbm.params.weights]
        for e,_ in enumerate(expectations):
            expectations[e] = np.dot(values[e].T, values[e+1]) / sample_num
        return DBM_params(dbm.layers, initial_params=(expectations))
    
    @classmethod
    def smci(cls, dbm, update_time=1000, sample_num=1000):
        if len(dbm.layers) != 3:
            raise TypeError("smci method only supports 3-layer DBM.")
        values = cls._sampling(dbm, update_time, sample_num)
        expectations = [None for i in dbm.params.weights]

        # dot(h1, w[0]) - h1 * w[0]
        under = (np.dot(values[1], dbm.params.weights[0].T)[:, :, np.newaxis] 
                    - values[1][:, np.newaxis, :] * dbm.params.weights[0][np.newaxis, :, :])
        # dot(h0, w[0]) - h0 * w[0] + dot(h2, w[1])
        upper = (np.dot(values[0], dbm.params.weights[0])[:, np.newaxis, :] 
                    - values[0][:, :, np.newaxis] * dbm.params.weights[0][np.newaxis, :, :]
                    + np.dot(values[2], dbm.params.weights[1].T)[:, np.newaxis, :] )
        expectations[0] = np.mean(tantan(under, upper, dbm.params.weights[0]), axis=0)

        # dot(h1, w[1]) - h1 * w[1]
        upper = (np.dot(values[1], dbm.params.weights[1])[:, np.newaxis, :] 
                    - values[1][:, :, np.newaxis] * dbm.params.weights[1][np.newaxis, :, :])
        # dot(h2, w[1]) - h2 * w[1] + dot(h0, w[0])
        under = (np.dot(values[2], dbm.params.weights[1].T)[:, :, np.newaxis] 
                    - values[2][:, np.newaxis, :] * dbm.params.weights[1][np.newaxis, :, :]
                    + np.dot(values[0], dbm.params.weights[0])[:, :, np.newaxis] )
        expectations[1] = np.mean(tantan(upper, under, dbm.params.weights[1]), axis=0) 

        return DBM_params(dbm.layers, initial_params=(expectations))

    # !!! this function takes exponential running time and requires HUGE memory !!!
    @classmethod
    def exact(cls, dbm):
        if len(dbm.layers) != 3:
            raise TypeError("exact method only supports 3-layer DBM.")

        expectations = [None for i in dbm.params.weights]
        bits = get_bits(np.max(dbm.layers))
        probability = dbm.probability()

        # lbits: [(2**i, i), (2**j, j), (2**k, k)]
        lbits = [bits[0:2**dbm.layers[i], 0:dbm.layers[i]] for i in range(len(dbm.layers))]
        # (2**i, 2**j, 2**k, i, j) -> (i, j)
        expectations[0] = np.sum(lbits[0][:, np.newaxis, np.newaxis, :, np.newaxis]
                            * lbits[1][np.newaxis, :, np.newaxis, np.newaxis, :]
                            * probability[:, :, :, np.newaxis, np.newaxis], axis=(0, 1, 2))
        # (2**i, 2**j, 2**k, j, k) -> (j, k)
        expectations[1] = np.sum(lbits[1][np.newaxis, :, np.newaxis, :, np.newaxis]
                            * lbits[2][np.newaxis, np.newaxis, :, np.newaxis, :]
                            * probability[:, :, :, np.newaxis, np.newaxis], axis=(0, 1, 2))

        return DBM_params(dbm.layers, initial_params=(expectations))