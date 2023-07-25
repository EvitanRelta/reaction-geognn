#!/bin/bash

DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$DGL_PYTHON_PATH train_reaction.py \
    --notes "finetune QM9 -> b97d3 -> wb97xd3" \
    --pretrained_chkpt_path "/home/tzongzhi/mygnn/lightning_logs/version_137/checkpoints/epoch=10-std_val_loss=4.42e-01.ckpt" \
