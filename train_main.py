#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from mltools.data import Data
from mltools.optimizer import SGD
from DBM import DBM

def main():
    dbm = DBM([5, 5, 5])
    data = Data(np.random.choice([-1, +1], (1000, 5)))
    dbm.train(data, 100, SGD({"weight0":1.0, "weight1":1.0}))

if __name__=="__main__":
    main()