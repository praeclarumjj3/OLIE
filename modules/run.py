# -*- coding: utf-8 -*-
from __future__ import print_function

from torch._C import device
from modules.dataloader import get_loader
import torch
import torch.nn.functional as F
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
        default=8,
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


def gram_matrix(inputs):
    grams = []
    for i in range(inputs.shape[0]):
        tensor = inputs[i]
        n_filters, h, w = tensor.size()
        tensor = tensor.view(n_filters, h * w)
        grams.append(torch.mm(tensor, tensor.t()))
  
    return torch.stack(grams,0)

layer_names = ['encoder.conv2', 
'decoder.conv1_2',
'decoder.convA2_3',
'decoder.conv4a',
'decoder.conv5a']

def normalize(inputs):
    # pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).cuda().view(3, 1, 1)
    pixel_std = torch.Tensor([57.375, 57.120, 58.395]).view(3, 1, 1).cuda()
    normalizer = lambda x: (x) / pixel_std
    return normalizer(inputs)

def get_features(image):

    feats = {}
    length = image.shape[0]
    x = []
    for i in range(length):
        x.append(image[i].squeeze(0))


    with torch.no_grad():
        masks, images = editor.solo(x)
        images = normalize(images)
        size = masks.shape[2]
        images = F.interpolate(images,(size,size))
        masks = F.sigmoid(masks)
        masks = torch.ones_like(masks) - masks
        x = torch.cat([masks,images], dim=1)
        for name, layer in editor.reconstructor.named_modules():
            if str(name) == "encoder" or str(name) == "decoder" or str(name) == "":
                continue
            x = layer(x)
            if str(name) in layer_names:
              feats[str(name)] = x
    
    return feats

def s_loss(inputs, targets):

    l1_loss = nn.L1Loss()

    style_loss = torch.tensor(0.).cuda()

    hole_features = get_features(inputs)
    recons_features = get_features(targets)

    for name in layer_names:
        style_loss += l1_loss(hole_features[name], recons_features[name])
    
    return style_loss


def recons_loss(outputs, images, hole_images, masks):
    loss = nn.L1Loss()
    inputs = torch.stack(images,0).cuda()
    hole_inputs = torch.stack(hole_images, 0)
    outputs = un_normalize(outputs)
    masks = torch.stack(masks,0).cuda()

    hole_loss =  loss(hole_inputs * masks, outputs * masks)

    style_mask = torch.tensor(1.) - masks
    style_loss = s_loss(inputs * style_mask, outputs * style_mask)

    alpha = torch.tensor(0.33).cuda()

    return (torch.tensor(1.0).cuda()-alpha)*hole_loss + alpha*style_loss


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

            inputs, hole_images, masks = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = recons_loss(outputs, inputs, hole_images, masks)
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
    
    if args.eval:
        coco_test_loader, _ = get_loader(device=device, \
                                    root=args.coco+'val2017', \
                                        json=args.coco+'annotations/instances_val2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=False, \
                                                    num_workers=0)
        editor_eval =Editor(solo,reconstructor)
        editor_eval.load_state_dict(torch.load(args.PATH))
        eval(editor_eval.to(device),coco_test_loader)
    else:
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        logger.info("Instantiating Editor")
        editor = Editor(solo,reconstructor)
        # for name, layer in editor.reconstructor.named_modules():
        #     print(name)
        # exit()
        if args.load:
            editor.load_state_dict(torch.load(args.PATH))
        editor.to(device)
        coco_train_loader, _ = get_loader(device=device, \
                                    root=args.coco+'train2017', \
                                        json=args.coco+'annotations/instances_train2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=False, \
                                                    num_workers=0)
        train(model=editor,num_epochs=args.num_epochs, dataloader=coco_train_loader)