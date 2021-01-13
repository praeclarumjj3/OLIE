#!/bin/sh

python modules/demo.py \
    --config-file configs/R50_3x.yaml \
    --input inputs/input1.jpg inputs/input2.jpg inputs/input3.jpeg\
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth