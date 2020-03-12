#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/w20/exact_exact/"`date +%Y-%m-%d_%H-%M-%S`""
mkdir -p $DIR

for i in `seq $1`
do
    ./train_main.py ./config/w20/555_exact_exact.json 2000 -d $DIR -s "exact_exact"
done
