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
from modules.reconstructor import Reconstructor, Encoder, BaseDecoder, EditDecoder, OrigDecoder
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm
import numpy as np
import warnings
from detectron2.utils.logger import setup_logger
from etaprogress.progress import ProgressBar
from detectron2.checkpoint import DetectionCheckpointer
import torchvision.models as models


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
        default=4,
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
    parser.add_argument(
        "--pretrain",
        help="To train the original reconstructor",
        default=False,
        type=bool
    )
    return parser


def un_normalize(inputs):
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(device).view(3, 1, 1)
    un_normalizer = lambda x: (x + pixel_mean) * pixel_std
    return un_normalizer(inputs)

def normalize(inputs):
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(device).view(3, 1, 1)
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    return normalizer(inputs)

def vgg_normalize(inputs):
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).cuda()
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).cuda()
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    return normalizer(inputs)


def get_features(image, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                #   '19': 'conv4_1',
                #   '28': 'conv5_1'
                  }
    features = {}
    image = torch.clamp(image, min=0., max = 255.) * torch.tensor(1./255)
    image = torch.stack([image[:,2,:,:],image[:,1,:,:],image[:,0,:,:]],1)

    x = vgg_normalize(image)
    for name, layer in enumerate(vgg.features):
      x = layer(x)
      if str(name) in layers:
        features[layers[str(name)]] = x
    
    return features

def s_loss(inputs, targets):
    
    l1_loss = nn.L1Loss()
    style_loss = torch.tensor(0.).cuda()
    hole_features = get_features(inputs)
    recons_features = get_features(targets)

    for layer in hole_features:
        style_loss += l1_loss(hole_features[layer], recons_features[layer])
    
    return style_loss

def p_loss():
    t_loss = torch.norm(editor_edit.reconstructor.decoder.perb4.W) + \
        torch.norm(editor_edit.reconstructor.decoder.perb4a.W) + \
            torch.norm(editor_edit.reconstructor.decoder.perb5.W) + \
                torch.norm(editor_edit.reconstructor.decoder.perb5a.W)
    
    return t_loss

def edit_loss(outputs, images, hole_images, masks):
    loss = nn.L1Loss()
    inputs = torch.stack(images,0).cuda()
    hole_inputs = torch.stack(hole_images, 0)
    outputs = un_normalize(outputs)
    masks = torch.stack(masks,0).cuda()

    hole_masks = torch.tensor(1.) - masks
    bg_loss =  loss(hole_inputs * hole_masks, outputs * hole_masks)
    
    hole_loss = s_loss(inputs * masks, outputs * masks)

    perb_loss = p_loss()

    alpha = torch.tensor(10., dtype=float)
    beta = torch.tensor(0.1, dtype=float)
    
    t_loss = bg_loss + alpha*hole_loss + beta*perb_loss

#     print('Style Loss: {}'.format(hole_loss))
#     print('Simple Loss: {}'.format(bg_loss))
#     print('Alpha: {}'.format(alpha))
#     print('Total Loss: {}'.format(t_loss))
#     print('----------------------')

    return t_loss

def recons_loss(outputs, images):
    loss = nn.L1Loss()
    inputs = torch.stack(images,0).cuda()
    outputs = un_normalize(outputs)

    bg_loss =  loss(inputs, outputs)
    styles_loss = s_loss(inputs,outputs)

    alpha = torch.tensor(10., dtype=float)

    t_loss = bg_loss + alpha*styles_loss

    return t_loss

def pretrain(model, num_epochs, dataloader):
    
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

            inputs, _, __ = data

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
            torch.save(base_decoder.state_dict(),'checkpoints/base_decoder.pth')
            torch.save(model.state_dict(),'checkpoints/editor_pretrained.pth')
            torch.save(encoder.state_dict(),'checkpoints/encoder.pth')
    logger.info("Finished Training with best loss: {} at Epoch: {}".format(best_loss, best_epoch))
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), epoch_loss)
    if not os.path.exists('losses/'):
            os.makedirs('losses/')
    plt.savefig('losses/pretrain_loss_{}.png'.format(args.lr))

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
            loss = edit_loss(outputs, inputs, hole_images, masks)
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
    
    encoder = Encoder(in_channels=r.shape[1])
    edit_decoder = EditDecoder()
    orig_decoder = OrigDecoder()
    base_decoder = BaseDecoder()
    orig_reconstructor = Reconstructor(encoder=encoder, decoder=orig_decoder, base_decoder=base_decoder)
    edit_reconstructor = Reconstructor(encoder=encoder, decoder=edit_decoder, base_decoder=base_decoder)
    # reconstructor = Reconstructor(in_channels=r.shape[1])

    edit_recons_params = sum(p.numel() for p in edit_reconstructor.parameters())
    solo_params = sum(p.numel() for p in solo.parameters())
    orig_decoder_params = sum(p.numel() for p in orig_decoder.parameters())

    logger.info("Total Params: {}".format(edit_recons_params+solo_params+orig_decoder_params))
    logger.info("Trainable Params: {}".format(edit_recons_params+orig_decoder_params))
    logger.info("Non-Trainable Params: {}".format(solo_params))
    
    if args.eval:
        coco_test_loader, _ = get_loader(device=device, \
                                    root=args.coco+'val2017', \
                                        json=args.coco+'annotations/instances_val2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=False, \
                                                    num_workers=0)
        editor_eval =Editor(solo,edit_reconstructor)
        editor_eval.load_state_dict(torch.load(args.PATH))
        eval(editor_eval.to(device),coco_test_loader)
    else:
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        logger.info("Instantiating Editor")
        editor_orig = Editor(solo,orig_reconstructor)
        editor_edit = Editor(solo,edit_reconstructor)
        # for name, layer in editor.reconstructor.named_modules():
        #     print(name)
        # exit()
        coco_train_loader, _ = get_loader(device=device, \
                                        root=args.coco+'train2017', \
                                            json=args.coco+'annotations/instances_train2017.json', \
                                                batch_size=args.batch_size, \
                                                    shuffle=False, \
                                                        num_workers=0)
        
        if args.pretrain:
            if args.load:
                editor_orig.load_state_dict(torch.load(args.PATH))
            editor_orig.to(device)

            vgg = models.vgg19(pretrained=True).cuda().eval()

            for i, layer in enumerate(vgg.features):
                if isinstance(layer, torch.nn.MaxPool2d):
                    vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

            pretrain(model=editor_orig,num_epochs=args.num_epochs, dataloader=coco_train_loader)
        
        else:
            if args.load:
                editor_edit.load_state_dict(torch.load(args.PATH))
            editor_edit.to(device)
            base_decoder.load_state_dict(torch.load('checkpoints/base_decoder.pth'))
            # encoder.load_state_dict(torch.load('checkpoints/encoder.pth'))
            vgg = models.vgg19(pretrained=True).cuda().eval()

            for i, layer in enumerate(vgg.features):
                if isinstance(layer, torch.nn.MaxPool2d):
                    vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

            train(model=editor_edit,num_epochs=args.num_epochs, dataloader=coco_train_loader)