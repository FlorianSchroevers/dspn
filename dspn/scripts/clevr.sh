#!/bin/bash


DATASET=$1
NUM=$2
USE_BASELINE=$3

if [ ! -z "$USE_BASELINE" ]
then
    PREFIX="base"
    ARGS="--baseline --decoder MLPDecoder"
else
    PREFIX="dspn"
    ARGS="--decoder DSPN"
fi

set -x

python train.py --show --loss chamfer --encoder RNFSEncoder --dim 512 --dataset $DATASET --epochs 1 --latent 512 --supervised --name $PREFIX-$DATASET-$NUM --iters 10 --lr 0.0003 --huber-repr 0.1 --mask-feature $ARGS

# python train.py --show --loss chamfer --encoder RNFSEncoder --dim 512 --dataset cats --epochs 10 --latent 512 --supervised --infer-name --iters 20 --lr 0.0003 --huber-repr 0.1 --mask-feature --decoder DSPN --num-workers 0 --full-eval

# python train.py --show --loss hungarian --encoder RNFSEncoder --dim 512 --dataset clevr-box --epochs 10 --latent 512 --supervised --name test --iters 10 --inner-lr 800 --resume logs/dspn-cats-2 --eval-only --export-dir out/clevr-box/dspn-cats-2-30  --decoder DSPN --mask-feature --export-progress --export-n 100 --n-workers 0