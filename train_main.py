#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import logging
import argparse
import json
import time
import mltools.optimizer
import os.path
from mltools.data import Data
from mltools import LearningLog
from DBM import DBM

parser = argparse.ArgumentParser("DBM learning script.", add_help=False)
parser.add_argument("learning_config", action="store", type=str, help="path of learning configuration file.")
parser.add_argument("learning_epoch", action="store", type=int, help="numbers of epochs.")
parser.add_argument("-l", "--log_level", action="store", type=str, default="INFO", help="learning log output level.")
parser.add_argument("-d", "--output_directory", action="store", type=str, default="./results/", help="directory to output parameter & log")
parser.add_argument("-s", "--filename_suffix", action="store", type=str, default=None, help="filename suffix")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : [%(levelname)s] %(message)s', level=getattr(logging, args.log_level))
np.seterr(over="raise", invalid="raise")

def main():
    logging.debug("Loading configuration file.")
    config = json.load(open(args.learning_config, "r"))

    logging.debug("Loading Initial/Generative models.")
    dbm = DBM.load(config["initial_model"], config["data_expectation"], config["model_expectation"])
    gen_dbm = DBM.load(config["generative_model"])

    logging.debug("Loading learning data.")
    learning_data = Data(np.load(config["learning_data"]))
    if "data_limit" in config:
        learning_data = Data(learning_data.data[:config["data_limit"]])

    logging.info("Optimizer: %s" % config["optimizer"])
    optimizer = getattr(mltools.optimizer, config["optimizer"])(**config["optimizer_args"])

    setting_log = {
        "learning_epoch": args.learning_epoch,
        "learning_configfile": args.learning_config,
        "traindata_size": len(learning_data),
    }
    setting_log.update(config)
    learning_log = LearningLog(setting_log)

    logging.info("Train started.")
    dbm.train(learning_data, args.learning_epoch, gen_dbm, learning_log, optimizer, minibatch_size=config["minibatch_size"], test_interval=config["test_interval"])
    logging.info("Train ended.")

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    suffix = "_"+args.filename_suffix if args.filename_suffix is not None else ""
    model_filepath = os.path.join(args.output_directory, "{}_model.json".format(timestamp+suffix))
    log_filepath = os.path.join(args.output_directory, "{}_log.json".format(timestamp+suffix))

    dbm.save(model_filepath)
    logging.info("Model parameters were dumped to: {}".format(model_filepath))
    learning_log.save(log_filepath)
    logging.info("Learning log was dumped to: {}".format(log_filepath))

if __name__=="__main__":
    main()