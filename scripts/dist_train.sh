#!/bin/sh

python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=29502 \
    train.py \
    --config-file configs/R50_3x.yaml \
    --opts MODEL.WEIGHTS SOLOv2_R50_3x.pth