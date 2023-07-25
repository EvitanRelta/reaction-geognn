#!/bin/bash

DGL_PYTHON_PATH="/home/tzongzhi/anaconda3/envs/dgl-geognn/bin/python"

$DGL_PYTHON_PATH train_reaction.py \
    --notes "finetune QM9 -> wb97xd3" \
    --pretrained_encoder_chkpt_path "TBC" \
