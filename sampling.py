#!/usr/bin/env python
# -*- coding:utf-8 -*-

from DBM import DBM
from marginal_functions import act
import numpy as np
import argparse

parser = argparse.ArgumentParser("DRBM sampling script", add_help=False)
parser.add_argument("-u", "--update_time", action="store", default=1000, type=int, help="update time.") 
parser.add_argument("filename", action="store", type=str, help="gen DBM filename.")
parser.add_argument("sampling_num", action="store", type=int, help="number of sampling.")
args = parser.parse_args()

def main():
    dbm = DBM.load(args.filename)
    sample_num = args.sampling_num

    values = [np.random.choice([-1, 1], (sample_num, i)) for i in dbm.layers]
    for i in range(args.update_time):
        act_prob = act( np.dot(dbm.params.weights[0], values[1].T).T )
        values[0] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[0]), 1, -1)

        for l in range(1, len(dbm.layers)-1):
            act_prob = act( np.dot( values[l-1], dbm.params.weights[l-1] ) + np.dot(dbm.params.weights[l], values[l+1].T).T )
            values[l] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[l]), 1, -1)

        act_prob = act( np.dot( values[-2], dbm.params.weights[-1] ) )
        values[-1] = np.where(act_prob > np.random.rand(sample_num, dbm.layers[-1]), 1, -1)
    
    np.save("sampling", values[0])

if __name__=='__main__':
    main()