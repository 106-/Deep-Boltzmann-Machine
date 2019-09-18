#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import logging
import argparse
import json
import mltools.optimizer
from mltools.data import Data
from DBM import DBM

parser = argparse.ArgumentParser("DRBM sampling script", add_help=False)
parser.add_argument("learning_config", action="store", type=str, help="path of learning configuration file.")
parser.add_argument("learning_epoch", action="store", type=int, help="numbers of epochs.")
parser.add_argument("-l", "--log_level", action="store", type=str, default="INFO", help="learning log output level.")
parser.add_argument("-o", "--optimizer", action="store", type=str, default="Adamax", help="parameter update method.")
parser.add_argument("-m", "--minibatch_size", action="store", type=int, default=100, help="minibatch size.")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : [%(levelname)s] %(message)s', level=getattr(logging, args.log_level))
np.seterr(over="raise", invalid="raise")

def main():
    config = json.load(open(args.learning_config, "r"))
    dbm = DBM.load(config["initial_model"])
    gen_dbm = DBM.load(config["generative_model"])
    learning_data = Data(np.load(config["learning_data"]))

    learning_time = int(args.learning_epoch * len(learning_data) / args.minibatch_size)
    optimizer = getattr(mltools.optimizer, args.optimizer)()

    dbm.train(learning_data, learning_time, optimizer, minibatch_size=args.minibatch_size)

if __name__=="__main__":
    main()