#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$CONDA_PYTHON_PATH "$SCRIPT_DIR/../train_reaction.py" \
    --notes "finetune QM9 -> wb97xd3" \
    --pretrained_encoder_chkpt_path "/home/tzongzhi/mygnn/lightning_logs/version_134/checkpoints/epoch=19-std_val_loss=1.20e-01.ckpt" \
