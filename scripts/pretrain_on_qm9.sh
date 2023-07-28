#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$DGL_PYTHON_PATH "$SCRIPT_DIR/../train.py" \
    --notes "pretrain -> QM9" \
    --early_stop \
