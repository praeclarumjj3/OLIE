#!/bin/sh

python train.py \
    --config-file configs/R50_3x.yaml \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth