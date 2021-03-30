# -*- coding: utf-8 -*-
from torch._C import device
from datasets.composite_data.coco_fg_loader import get_loader
from datasets.composite_data.places2_loader import get_places2_loader
import torch
import argparse
import warnings
from etaprogress.progress import ProgressBar
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description="SOLOv2 Editor")
    parser.add_argument(
        "--coco",
        help="Path of coco dataset",
        default='datasets/coco/'
    )
    parser.add_argument(
        "--batch_size",
        help="Batch Size for dataloaders",
        default=1,
        type=int
    )
    parser.add_argument(
        "--places2",
        help="Path of places dataset",
        default='datasets/places365_standard/places2'
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    
    if not os.path.exists('datasets/bg_composite/'):
            os.makedirs('datasets/bg_composite/')
    
    if not os.path.exists('datasets/bg/'):
            os.makedirs('datasets/bg/')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    coco_loader, _ = get_loader(device=device, \
                                    root=args.coco+'train2017', \
                                        json=args.coco+'annotations/instances_train2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=False, \
                                                    num_workers=0)
    
    places_loader = get_places2_loader(root=args.places2, batch_size=1)

    total = len(coco_loader)
    bar = ProgressBar(total, max_width=80)
    for i, data in tqdm(enumerate(zip(coco_loader,places_loader), 0)):
        bar.numerator = i+1
        print(bar, end='\r')
        
        coco_data, places2_data = data

        coco_input, fg_mask, path = coco_data
        bg_input, label = places2_data
        
        bg_input = bg_input.squeeze(0)
        bg_input = torch.round(bg_input.cuda() * torch.tensor(255.))
        
        coco_input = coco_input[0]
        fg_mask = fg_mask[0]
        path = path[0]

        composite_image = bg_input.cuda() * (1-fg_mask) + coco_input * fg_mask

        composite_image = composite_image

        composite_image = composite_image.cpu() 
        composite_image = composite_image.permute(1, 2, 0).numpy()

        bg_input = bg_input.cpu() 
        bg_input = bg_input.permute(1, 2, 0).numpy()
        
        plt.imsave('datasets/bg_composite/'+path,composite_image.astype(np.uint8))
        plt.imsave('datasets/bg/'+path,bg_input.astype(np.uint8))
        
