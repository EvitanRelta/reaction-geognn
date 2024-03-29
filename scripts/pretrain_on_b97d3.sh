#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$CONDA_PYTHON_PATH "$SCRIPT_DIR/../train_reaction.py" \
    --notes "pretrain -> b97d3" \
    --dataset b97d3 \
    --early_stop \
