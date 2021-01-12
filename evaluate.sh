#!/bin/sh

python modules/run.py \
    --config-file configs/R50_3x.yaml \
    --eval True \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth