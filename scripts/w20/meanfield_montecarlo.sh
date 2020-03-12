#!/bin/sh

if [ $# -ne 1 ]; then
  echo "requires 1 arguments." 1>&2
  exit 1
fi

DIR="./results/w20/meanfield_montecarlo/"`date +%Y-%m-%d_%H-%M-%S`""
mkdir -p $DIR

for i in `seq $1`
do
    ./train_main.py ./config/w20/555_meanfield_montecarlo.json 2000 -d $DIR -s "meanfield_montecarlo"
done
