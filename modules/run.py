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
import multiprocessing as mp
import os
from tqdm import tqdm
import numpy as np
import warnings
from detectron2.utils.logger import setup_logger
from etaprogress.progress import ProgressBar

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
        "--eval",
        help="To eval or not",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--PATH",
        help="Path of the saved editor",
        default='checkpoints.editor.pth',
        type=str
    )
    return parser


def recons_loss(outputs, images):
    total_loss = float(0)
    loss = nn.L1Loss()
    edited_images = []

    for j in range(outputs.shape[0]):
        edited_images.append(outputs[j].squeeze(0))

    for o,i in zip(edited_images, images):
        total_loss += loss(i,o)
    return total_loss


def train(model, num_epochs, dataloader):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, betas=(0.9, 0.999), weight_decay=1e-2)

    best_loss = 1e10
    best_epoch = 0
    epoch_loss = []

    logger.info("Starting Training")
    for j in range(num_epochs):
        running_loss = []
        total = len(dataloader)
        bar = ProgressBar(total, max_width=60)
        for i, data in tqdm(enumerate(dataloader, 0)):
            bar.numerator = i
            print(bar, end='\r')
            # get the inputs; data is a list of [images]
            inputs = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = recons_loss(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            sys.stdout.flush()
        
        avg_loss = np.mean(running_loss)
        print("Epoch {}: Loss: {}".format(j+1,avg_loss))
        epoch_loss.append(avg_loss)
        
        if running_loss < best_loss:
            best_loss = running_loss
            best_epoch = j+1
            print('Model saved at Epoch: {}'.format(j+1))
            torch.save(model.state_dict(),'checkpoints/editor.pth')
    logger.info("Finished Training with best loss: {} at Epoch: {}".format(best_loss, best_epoch))
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), epoch_loss)
    plt.savefig('train_loss.png')

def eval(model, dataloader):
    
    model.eval()

    running_loss = []
    total = len(dataloader)
    bar = ProgressBar(total, max_width=60)
    logger.info("Starting Evaluation")
    for i, data in tqdm(enumerate(dataloader, 0)):
        bar.numerator = i
        print(bar, end='\r')
        # get the inputs; data is a list of [images]
        inputs = data
        outputs = model(inputs)
        loss = recons_loss(outputs, inputs)
        
        running_loss.append(loss.item())
        sys.stdout.flush()
        
    avg_loss = np.mean(running_loss)
    print("Eval Loss: {}".format(avg_loss))
        
    plt.plot(np.linspace(1, total, total).astype(int), running_loss)
    plt.savefig('eval_loss.png')

if __name__ == "__main__":
    if not os.path.exists('checkpoints/'):
        os.makedirs('checkpoints/')
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()
    args = get_parser().parse_args()
    logger.info("Arguments: " + str(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = setup_cfg(args)

    solo = SOLOv2(cfg=cfg).to(device)
    for param in solo.parameters():
        param.requires_grad = False
    
    image = torch.randint(0, 256, (3,64,64))
    batched_input = []
    batched_input.append(image)
    r,_ = solo(batched_input)

    reconstructor = Reconstructor(in_channels=r.shape[1])
    
    coco_train_loader = get_loader(device=device, \
                                    root=args.coco+'train2017', \
                                        json=args.coco+'annotations/instances_train2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=True, \
                                                    num_workers=8)

    coco_test_loader = get_loader(device=device, \
                                    root=args.coco+'val2017', \
                                        json=args.coco+'annotations/instances_val2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=False, \
                                                    num_workers=8)
    
    if args.eval:
        editor_eval =Editor(solo,reconstructor)
        editor_eval.load_state_dict(torch.load(args.PATH))
        eval(editor_eval.to(device),coco_test_loader)
    else:
        logger.info("Instantiating Editor")
        editor = Editor(solo,reconstructor)
        train(model=editor.to(device),num_epochs=args.num_epochs, dataloader=coco_train_loader)
    
#     total = len(coco_test_loader)
#     bar = ProgressBar(total, max_width=60)
#     for i, data in enumerate(coco_test_loader, 0):
#         bar.numerator = i
#         print(bar, end='\r')
#         # get the inputs; data is a list of [inputs, labels]
#         inputs = data
#         results=solo(inputs[:2])
#         sys.stdout.flush()