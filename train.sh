#!/bin/sh

python modules/run.py \
    --config-file configs/R50_3x.yaml \
    --PATH 'checkpoints/editor.pth' \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth