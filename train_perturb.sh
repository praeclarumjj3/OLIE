#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python modules/perturbor_code/run.py \
    --config-file configs/R50_3x.yaml \
    --lr 1e-3 \
    --num_epochs 30 \
    --batch_size 8 \
    --PATH 'checkpoints/editor_pertutb.pth' \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth