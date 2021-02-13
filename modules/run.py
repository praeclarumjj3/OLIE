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
from loss import ReconLoss, VGGLoss


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
    return parser

def visualize(x,y,z):
    x = x[0].cpu() 
    x = x.permute(1, 2, 0).numpy()
    y = y[0].cpu() 
    y = y.permute(1, 2, 0).numpy()
    z = z[0].cpu() 
    z = z.permute(1, 2, 0).numpy()
    f, (ax1,ax2,ax3) = plt.subplots(1,3)
    x = x[:,:,::-1]
    y = y[:,:,::-1]
    z = z[:,:,::-1]
    ax1.imshow(x)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(y)
    ax2.set_title("Masked Image")
    ax2.axis('off')

    ax3.imshow(z)
    ax3.set_title("Reconstruction")
    ax3.axis('off')

    f.savefig('visualizations/run.jpg')


def normalize(inputs):
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1).cuda()
    pixel_std = torch.Tensor([57.375, 57.120, 58.395]).view(3, 1, 1).cuda()
    un_normalizer = lambda x: (x - pixel_mean) / pixel_std
    return un_normalizer(inputs)

def un_normalize(inputs):
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1).cuda()
    pixel_std = torch.Tensor([57.375, 57.120, 58.395]).view(3, 1, 1).cuda()
    un_normalizer = lambda x: x * pixel_std + pixel_mean
    return un_normalizer(inputs)

def vgg_normalize(inputs):
    pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).cuda()
    pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).cuda()
    normalizer = lambda x: (x - pixel_mean) / pixel_std
    return normalizer(inputs)

def vgg_preprocess(image):
    image = image * torch.tensor(1./255)
    image = torch.stack([image[:,2,:,:],image[:,1,:,:],image[:,0,:,:]],1)

    image = vgg_normalize(image)
    return image


def s_loss(targets, recons, masks):

    targets = vgg_preprocess(targets)
    recons = vgg_preprocess(recons)

    style_loss = vgg_loss(recons, targets, masks)
    
    return style_loss


def edit_loss(outputs, images, hole_images, masks):
    inputs = torch.stack(images,0).cuda()
    # hole_inputs = torch.stack(hole_images, 0)
    outputs = un_normalize(outputs)
    # masks = torch.stack(masks,0).cuda()

    # hole_masks = torch.tensor(1.) - masks

    bg_loss =  recon_loss(outputs, inputs)

    
    # hole_loss = s_loss(hole_inputs, outputs, masks)

    # alpha = torch.tensor(50., dtype=float)
    t_loss = bg_loss

#     print('Style Loss: {}'.format(hole_loss))
#     print('Simple Loss: {}'.format(bg_loss))
#     print('Total Loss: {}'.format(t_loss))
#     print('----------------------')

    return t_loss

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
            
            if i%30 == 0:
                print('Loss: {}'.format(loss.item()))
                inputs = torch.stack(inputs,0).cuda()
                outputs = un_normalize(outputs)
                hole_inputs = torch.stack(hole_images, 0)
                masks = torch.stack(masks,0).cuda()
                visualize(inputs*torch.tensor(1./255), hole_inputs*torch.tensor(1./255),torch.round(outputs.detach())*torch.tensor(1./255))
            
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
    plt.plot(np.linspace(1, num_epochs, num_import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

def masking(image, phase, index):
    overlap = torch.ones_like(image)
    if phase=="single":
        for i in range(int(overlap.shape[0])):
            overlap[i][index] = torch.tensor(0, dtype = float)
    elif phase=="mult":
        for i in range(overlap.shape[0]):
            overlap[i][index[0]:index[1]] = torch.tensor(0, dtype = float)
    
    return image*overlap

def normalize(inputs):
    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(3, 1, 1).cuda()
    pixel_std = torch.Tensor([57.375, 57.120, 58.395]).view(3, 1, 1).cuda()
    normalizer = lambda x: (x.cuda() - pixel_mean) / pixel_std
    return normalizer(inputs)

def visualize(x,layer):
    dim = int(x.shape[1])
    x = x[0].cpu() 
    x = x.permute(1, 2, 0).numpy()
    f, axarr = plt.subplots(int(dim**0.5),int(dim**0.5),figsize=(16,16))
    for j in range(x.shape[2]):
        r = int(j/dim**0.5)
        c = int(j%dim**0.5)
        axarr[r,c].imshow(x[:,:,j])
        axarr[r,c].axis('off')
    f.savefig('visualizations/{}.jpg'.format(layer))

class Reconstructor(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()

    def forward(self, masks, imgs):
        size = masks.shape[2]
        images = F.interpolate(imgs,(size,size))
        images = normalize(images)
        # masks = F.tanh(masks)
        # masks = torch.ones_like(masks) - masks
        # masks = masks ** 2
        # inputs = torch.cat([masks,images], dim=1)
        enc_masks = self.encoder(images,masks)
        # images = F.interpolate(imgs,(size*4,size*4))
        x = self.decoder(enc_masks)

        return x

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(512)

        self.mask_conv1 = nn.Conv2d(in_channels,256,kernel_size=1,stride=1)
        self.mask_conv2 = nn.Conv2d(256,512,kernel_size=1,stride=1)
        
    def forward(self, masks):
        masks = self.mask_conv1(masks)
        masks = F.leaky_relu(masks, negative_slope=0.4)
        masks = self.mask_conv2(masks)
        masks = F.leaky_relu(masks, negative_slope=0.4)

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = F.leaky_relu(x, negative_slope=0.4)

        return masks

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.conv1_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(256)
        # self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        
        # # dilated convolution blocks
        # self.convA2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8)
        # self.convA2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.convA2_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        

        # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1) 
        # self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.conv5a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
        # self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        # self.conv7 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.mask_conv1 = nn.Conv2d(512,256,kernel_size=1,stride=1)
        self.mask_conv2 = nn.Conv2d(256,64,kernel_size=1,stride=1)
        self.mask_conv3 = nn.Conv2d(64,32,kernel_size=1,stride=1)
        self.mask_conv4 = nn.Conv2d(32,3,kernel_size=1,stride=1)


    def forward(self, x, masks, images):
        # x = self.conv1_1(x)
        # x = self.bn1(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv1_2(x)
        # x = self.bn2(x)
        # x = F.leaky_relu(x, negative_slope=0.4)

        # x = self.convA2_1(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.convA2_2(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.convA2_3(x)
        # x = F.leaky_relu(x, negative_slope=0.4)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv4(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv4a(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        # x = self.conv5(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv5a(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        # x = self.conv6(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv7(x)

        masks = self.mask_conv1(masks)
        masks = F.leaky_relu(masks, negative_slope=0.4)
        masks = self.mask_conv2(masks)
        masks = F.leaky_relu(masks, negative_slope=0.4)
        masks = F.upsample(masks, scale_factor=4, mode='bilinear', align_corners=False)
        masks = self.mask_conv3(masks)
        masks = F.leaky_relu(masks, negative_slope=0.4)
        masks = self.mask_conv4(masks)
        
        return masks
    total = len(dataloader)
    bar = ProgressBar(total, max_width=80)
    logger.info("Starting Evaluation")
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            bar.numerator = i+1
            print(bar, end='\r')
            
            inputs = data
            outputs = model(inputs)
            loss = edit_loss(outputs, inputs)

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

    edit_recons_params = sum(p.numel() for p in reconstructor.parameters())
    solo_params = sum(p.numel() for p in solo.parameters())

    logger.info("Total Params: {}".format(edit_recons_params+solo_params))
    logger.info("Trainable Params: {}".format(edit_recons_params))
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

        coco_train_loader, _ = get_loader(device=device, \
                                        root=args.coco+'train2017', \
                                            json=args.coco+'annotations/instances_train2017.json', \
                                                batch_size=args.batch_size, \
                                                    shuffle=True, \
                                                        num_workers=0)
        
        if args.load:
            editor.load_state_dict(torch.load(args.PATH))
        
        editor.to(device)
        
        vgg_loss = VGGLoss(masked=True)
        recon_loss = ReconLoss(masked=True)
        
        train(model=editor,num_epochs=args.num_epochs, dataloader=coco_train_loader)