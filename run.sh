#!/bin/bash

DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"
DATASET_SIZE=15000 # must be multiples of batch size.
BATCH_SIZE=2000
GPU=1

NUM_BATCHES=$((DATASET_SIZE / BATCH_SIZE))

$DGL_PYTHON_PATH train_reaction.py \
    --dropout-rate 0 \
    --epochs 1000 \
    --lr 1e-4 \
    \
    --gnn-layers 3 \
    --embed-dim 256 \
    \
    --batch-size $BATCH_SIZE \
    --overfit-batches $NUM_BATCHES \
    --device "cuda:$GPU" \
