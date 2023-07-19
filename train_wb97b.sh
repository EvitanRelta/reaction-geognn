#!/bin/bash

DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"
GPU=0

$DGL_PYTHON_PATH train_reaction.py \
    --epochs 1000 \
    \
    --device "cuda:$GPU" \
    --notes "pretrain b97, pool concat-reactant-diff, GNN-aggregate bonds b4 pooling" \
