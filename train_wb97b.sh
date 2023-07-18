#!/bin/bash

DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"
GPU=1

$DGL_PYTHON_PATH train_reaction.py \
    --dropout_rate 0.1 \
    --epochs 1000 \
    --lr 3e-4 \
    \
    --gnn_layers 3 \
    --embed_dim 256 \
    \
    --batch_size 50 \
    --device "cuda:$GPU" \
    --shuffle \
