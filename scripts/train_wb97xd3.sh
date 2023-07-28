#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$CONDA_PYTHON_PATH "$SCRIPT_DIR/../train_reaction.py" \
    --notes "no pretraining" \
