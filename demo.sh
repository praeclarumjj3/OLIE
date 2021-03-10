#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python modules/demo.py \
    --config-file configs/R50_3x.yaml \
    --PATH 'checkpoints/editor.pth' \
    --input inputs/val/val1.jpg inputs/val/val2.jpg inputs/val/val3.jpg inputs/val/val4.jpg \
    --kernel_visualize True \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth