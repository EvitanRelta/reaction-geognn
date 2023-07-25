#!/bin/bash

DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$DGL_PYTHON_PATH train_reaction.py \
    --notes "pretrain b97" \
    --dataset b97d3 \
    --early_stop \
