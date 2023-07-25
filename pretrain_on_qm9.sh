#!/bin/bash

DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$DGL_PYTHON_PATH train.py \
    --notes "QM9 5e-5" \
    --lr 5e-5 \
    --epochs 100 \
