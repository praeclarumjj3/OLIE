#!/bin/sh

python baselines/profill_on_solo/profill_solo.py \
    --config-file configs/R50_3x.yaml \
    --input baselines/imgs/0.jpg baselines/imgs/1.jpg baselines/imgs/2.jpg baselines/imgs/3.jpg baselines/imgs/4.jpg\
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth