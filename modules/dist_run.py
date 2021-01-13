# -*- coding: utf-8 -*-
from __future__ import print_function

from torch._C import device
from modules.dataloader import get_loader
import torch
from torch import nn
import sys
from adet.config import get_cfg
from modules.solov2 import SOLOv2
from modules.reconstructor import Reconstructor
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import numpy as np
import warnings
from detectron2.utils.logger import setup_logger
from etaprogress.progress import ProgressBar
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from detectron2.checkpoint import DetectionCheckpointer

warnings.filterwarnings("ignore")

class Editor(nn.Module):
    
    def __init__(self, solo, reconstructor):
        super().__init__()

        # get the device of the model
        self.solo = solo
        self.reconstructor = reconstructor

    def forward(self, x):
        results, images = self.solo(x)
        output = self.reconstructor(results, images)
        return output


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="SOLOv2 Editor")
    parser.add_argument(
        "--config-file",
        default="configs/R50_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated test images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--coco",
        help="Path of coco dataset",
        default='datasets/coco/'
    )
    parser.add_argument(
        "--batch_size",
        help="Batch Size for dataloaders",
        default=16,
        type=int
    )
    parser.add_argument(
        "--num_epochs",
        help="Epochs",
        default=30,
        type=int
    )

    parser.add_argument(
        "--local_rank",
        help="local gpu id",
        default=0,
        type=int
    )
    return parser


def recons_loss(outputs, images):
    loss = nn.L1Loss()
    inputs = torch.stack(images,0).cuda()
    return loss(inputs,outputs)

def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


if __name__ == "__main__":
    
    args = get_parser().parse_args()
    logger = setup_logger()
    
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    global_rank = dist.get_rank()
    world_size = torch.cuda.device_count()

    if global_rank == 0:
        logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    solo = SOLOv2(cfg=cfg)
    checkpointer = DetectionCheckpointer(solo)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    for param in solo.parameters():
        param.requires_grad = False
    
    image = torch.randint(0, 256, (3,64,64))
    batched_input = []
    batched_input.append(image)
    r,_ = solo(batched_input)

    reconstructor = Reconstructor(in_channels=r.shape[1])
    
    coco_test_loader, coco_test_set = get_loader(device=None,root=args.coco+'val2017', \
                                        json=args.coco+'annotations/instances_val2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=False, \
                                                    num_workers=0)

    # coco_train_loader, coco_train_set = get_loader(device=None, root=args.coco+'train2017', \
    #                                                 json=args.coco+'annotations/instances_train2017.json', \
    #                                                     batch_size=args.batch_size, \
    #                                                         shuffle=True, \
    #                                                             num_workers=0)
    if global_rank == 0:
        logger.info("Instantiating Editor")
    editor = Editor(solo,reconstructor)
    editor.cuda()
   
    editor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(editor)
    editor = DDP(editor, device_ids=[args.local_rank], output_device=args.local_rank)
    
    sampler = DistributedSampler(coco_test_set)
    optimizer = torch.optim.Adam(editor.parameters(), lr=1e-1, betas=(0.9, 0.999), weight_decay=1e-2)

    best_loss = 1e10
    best_epoch = 0
    epoch_loss = []

    if global_rank == 0:
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        logger.info("Starting Training")
    
    editor.train()
    for j in range(args.num_epochs):
        sampler.set_epoch(j)
        running_loss = []
        total = len(coco_test_loader)
        bar = ProgressBar(total, max_width=60)
        for i, data in tqdm(enumerate(coco_test_loader, 0)):
            if global_rank == 0:
                bar.numerator = i + 1
                print(bar, end='\r')

            for da in data:
                for d in da:
                    d.cuda()

            inputs = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = editor(inputs)
            loss = recons_loss(outputs, inputs)
            loss.backward()
            optimizer.step()
            reduce_loss(loss, global_rank, world_size)
            running_loss.append(loss.item())
            sys.stdout.flush()
        
        avg_loss = np.mean(running_loss)
        if global_rank == 0:
            print("Epoch {}: Loss: {}".format(j+1,avg_loss))
            epoch_loss.append(avg_loss)
        
        if avg_loss < best_loss and global_rank == 0:
            best_loss = avg_loss
            best_epoch = j+1
            print('Model saved at Epoch: {}'.format(j+1))
            torch.save(editor.state_dict(),'checkpoints/editor.pth')
    dist.destroy_process_group()
    logger.info("Finished Training with best loss: {} at Epoch: {}".format(best_loss, best_epoch))
    plt.plot(np.linspace(1, args.num_epochs, args.num_epochs).astype(int), epoch_loss)
    plt.savefig('train_loss.png')

