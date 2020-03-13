#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/w10/"`date +%Y-%m-%d_%H-%M-%S`"_smci_smci/"
mkdir -p $DIR

for i in `seq $1`
do
    ./train_main.py ./config/w10/555_smci_smci.json 5000 -d $DIR -s "smci_smci"
done
