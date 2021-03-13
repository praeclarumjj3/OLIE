# -*- coding: utf-8 -*-
from torch._C import device
from modules.coco_fg_loader import get_loader
from modules.places2_loader import get_places2_loader
import torch
import argparse
import warnings
from etaprogress.progress import ProgressBar
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

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
        default='datasets/places2/'
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    coco_loader, _ = get_loader(device=device, \
                                    root=args.coco+'train2017', \
                                        json=args.coco+'annotations/instances_train2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=False, \
                                                    num_workers=0)
    
    places_loader = get_places2_loader(isTrain=True)


    total = len(coco_loader)
    bar = ProgressBar(total, max_width=80)
    for i, coco_data, places2_data in tqdm(enumerate(zip(coco_loader,places_loader), 0)):
        bar.numerator = i+1
        print(bar, end='\r')

        coco_input, fg_mask = coco_data
        bg_input, _, path = places2_data

        composite_image = bg_input.cuda() * (1-fg_mask) + coco_input * fg_mask

        composite_image = composite_image*torch.tensor(1./255)

        print(composite_image.shape)
        composite_image.squeeze(0)

        composite_image = composite_image.cpu() 
        composite_image = composite_image.permute(1, 2, 0).numpy()
        
        plt.imsave('datasets/composite/'+path[0],composite_image.astype(np.uint8))
        
