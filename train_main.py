#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import logging
import argparse
import json
import mltools.optimizer
from mltools.data import Data
from mltools import LearningLog
from DBM import DBM

parser = argparse.ArgumentParser("DRBM sampling script", add_help=False)
parser.add_argument("learning_config", action="store", type=str, help="path of learning configuration file.")
parser.add_argument("learning_epoch", action="store", type=int, help="numbers of epochs.")
parser.add_argument("-l", "--log_level", action="store", type=str, default="INFO", help="learning log output level.")
parser.add_argument("-o", "--optimizer", action="store", type=str, default="Adamax", help="parameter update method.")
parser.add_argument("-m", "--minibatch_size", action="store", type=int, default=100, help="minibatch size.")
parser.add_argument("-t", "--test_interval", action="store", type=float, default=1.0, help="test interval(epoch).")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : [%(levelname)s] %(message)s', level=getattr(logging, args.log_level))
np.seterr(over="raise", invalid="raise")

def main():
    logging.debug("Loading configuration file.")
    config = json.load(open(args.learning_config, "r"))

    logging.debug("Loading Initial/Generative models.")
    dbm = DBM.load(config["initial_model"])
    gen_dbm = DBM.load(config["generative_model"])

    logging.debug("Loading learning data.")
    learning_data = Data(np.load(config["learning_data"]))

    logging.info("Optimizer: %s" % args.optimizer)
    optimizer = getattr(mltools.optimizer, args.optimizer)()

    setting_log = {
        "learning_epoch": args.learning_epoch,
        "learning_configfile": args.learning_config,
        "test_interval": args.test_interval,
        "minibatch_size": args.minibatch_size,
        "traindata_size": len(learning_data),
        "optimizer": args.optimizer,
    }
    setting_log.update(config)
    learning_log = LearningLog(setting_log)

    logging.info("Train started.")
    dbm.train(learning_data, args.learning_epoch, gen_dbm, learning_log, optimizer, minibatch_size=args.minibatch_size, test_interval=args.test_interval)
    logging.info("Train ended.")

if __name__=="__main__":
    main()