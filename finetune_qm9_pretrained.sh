#!/bin/bash

DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$DGL_PYTHON_PATH train_reaction.py \
    --notes "finetune qm9 -> wb97" \
    --epochs 100 \
    --pretrained_encoder_chkpt_path "/home/tzongzhi/mygnn/lightning_logs/version_115/checkpoints/qm9-epoch=76-std_val_loss=0.10.ckpt" \
