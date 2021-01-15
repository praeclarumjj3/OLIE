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
from detectron2.checkpoint import DetectionCheckpointer


warnings.filterwarnings("ignore")

class Editor(nn.Module):
    
    def __init__(self, solo, reconstructor):
        super().__init__()

        # get the device of the model
        self.solo = solo
        self.reconstructor = reconstructor

    def forward(self, x):
        masks, images = self.solo(x)
#         cut_masks = torch.cat([masks[:,:36,:,:],masks[:,36:72,:,:],masks[:,108:144,:,:]],dim=1)
        output = self.reconstructor(masks, images)
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
        "--lr",
        help="Learning Rate",
        default=1e-3,
        type=float
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
        default='checkpoints/editor.pth',
        type=str
    )
    parser.add_argument(
        "--load",
        help="To load pretrained weights for further training",
        default=False,
        type=bool
    )
    return parser


def un_normalize(inputs):
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(device).view(3, 1, 1)
    un_normalizer = lambda x: (x + pixel_mean) * pixel_std
    return un_normalizer(inputs)

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1','5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content layer
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in enumerate(model.features):
      x = layer(x)
      if str(name) in layers:
        features[layers[str(name)]] = x
    
    return features


def gram_matrix(inputs):
    grams = []
    for i in range(inputs.shape[0]):
        tensor = inputs[i]
        n_filters, h, w = tensor.size()
        tensor = tensor.view(n_filters, h * w)
        grams.append(torch.mm(tensor, tensor.t()))
  
    return torch.stack(grams,0)


def s_loss(inputs, outputs):

    in_gram = gram_matrix(inputs)
    out_gram = gram_matrix(outputs)
    loss = nn.MSELoss()
    style_loss = loss(out_gram, in_gram)
    return style_loss


def recons_loss(outputs, images):
    loss = nn.L1Loss()
    inputs = torch.stack(images,0).cuda()
    outputs = un_normalize(outputs)
    content_loss =  loss(inputs, outputs)
    style_loss = s_loss(inputs, outputs)

    return content_loss + style_loss


def train(model, num_epochs, dataloader):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

    best_loss = 1e10
    best_epoch = 0
    epoch_loss = []

    logger.info("Starting Training")
    for j in range(num_epochs):
        running_loss = []
        total = len(dataloader)
        bar = ProgressBar(total, max_width=80)
        for i, data in tqdm(enumerate(dataloader, 0)):
            bar.numerator = i+1
            print(bar, end='\r')

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
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = j+1
            print('Model saved at Epoch: {}'.format(j+1))
            torch.save(model.state_dict(),args.PATH)
    logger.info("Finished Training with best loss: {} at Epoch: {}".format(best_loss, best_epoch))
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), epoch_loss)
    if not os.path.exists('losses/'):
            os.makedirs('losses/')
    plt.savefig('losses/train_loss_{}.png'.format(args.lr))


def eval(model, dataloader):
    
    model.eval()

    running_loss = []
    total = len(dataloader)
    bar = ProgressBar(total, max_width=80)
    logger.info("Starting Evaluation")
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            bar.numerator = i+1
            print(bar, end='\r')
            
            inputs = data
            outputs = model(inputs)
            loss = recons_loss(outputs, inputs)

            running_loss.append(loss.item())
            sys.stdout.flush()
        
    avg_loss = np.mean(running_loss)
    print("Eval Loss: {}".format(avg_loss))
        
    plt.plot(np.linspace(1, total, total).astype(int), running_loss)
    if not os.path.exists('losses/'):
            os.makedirs('losses/')
    plt.savefig('losses/eval_loss_{}.png'.format(args.lr))


if __name__ == "__main__":
    logger = setup_logger()
    args = get_parser().parse_args()
    logger.info("Arguments: " + str(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = setup_cfg(args)

    solo = SOLOv2(cfg=cfg).to(device)
    checkpointer = DetectionCheckpointer(solo)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    for param in solo.parameters():
        param.requires_grad = False
    
    image = torch.rand(3,64,64)
    batched_input = []
    batched_input.append(image)
    r,_ = solo(batched_input)
    
    reconstructor = Reconstructor(in_channels=r.shape[1])

    recons_params = sum(p.numel() for p in reconstructor.parameters())
    solo_params = sum(p.numel() for p in solo.parameters())

    logger.info("Total Params: {}".format(recons_params+solo_params))
    logger.info("Trainable Params: {}".format(recons_params))
    logger.info("Non-Trainable Params: {}".format(solo_params))
    
    coco_train_loader, _ = get_loader(device=device, \
                                    root=args.coco+'train2017', \
                                        json=args.coco+'annotations/instances_train2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=True, \
                                                    num_workers=0)

    coco_test_loader, _ = get_loader(device=device, \
                                    root=args.coco+'val2017', \
                                        json=args.coco+'annotations/instances_val2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=True, \
                                                    num_workers=0)
    
    if args.eval:
        editor_eval =Editor(solo,reconstructor)
        editor_eval.load_state_dict(torch.load(args.PATH))
        eval(editor_eval.to(device),coco_test_loader)
    else:
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
            logger.info("Instantiating Editor")
        editor = Editor(solo,reconstructor)
        if args.load:
            editor.load_state_dict(torch.load(args.PATH))
        train(model=editor.to(device),num_epochs=args.num_epochs, dataloader=coco_train_loader)