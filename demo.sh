#!/bin/sh

python modules/run.py \
    --config-file configs/R50_3x.yaml \
    --input input1.jpg \
    --demo True \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth