#!/bin/bash

DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$DGL_PYTHON_PATH train_reaction.py \
    --notes "pretrain qm9 (epoch 25) -> b97" \
    --dataset b97d3 \
    --early_stop \
    --pretrained_encoder_chkpt_path "/home/tzongzhi/mygnn/lightning_logs/version_125/checkpoints/epoch=25-std_val_loss=1.18e-01.ckpt" \
