#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python modules/demo.py \
    --config-file configs/R50_3x.yaml \
    --PATH 'checkpoints/editor.pth' \
    --input inputs/val1.jpg inputs/val2.jpg inputs/val3.jpg inputs/val4.jpg \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth