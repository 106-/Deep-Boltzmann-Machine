#!/usr/bin/env python

import argparse
import json
from mltools import LogSet

def main():
    parser = argparse.ArgumentParser(description="aggregates learning logs.")
    parser.add_argument("datasource", action="store", type=str, help="log file source.  (if a directory is specified, search log files recursively.)")
    parser.add_argument("setting_file", action="store", type=str, help="arregation setting file. (must be json format.)")
    args = parser.parse_args()

    settings = json.load(open(args.setting_file, "r"))
    logset = LogSet(args.datasource, settings)
    logset.summary().plot(settings)

if __name__=='__main__':
    main()