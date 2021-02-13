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
from inpaint import Inpainter

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
        "--RECONS_PATH",
        help="Path of the saved reconstructor",
        default='checkpoints/editor_success.pth',
        type=str
    )
    parser.add_argument(
        "--PATH",
        help="Path of the saved inpainting model",
        default='checkpoints/inpainter.pth',
        type=str
    )
    parser.add_argument(
        "--load",
        help="To load pretrained weights for further training",
        default=False,
        type=bool
    )
    return parser

def visualize(x,y,z,w,v):
    x = x[0].cpu() 
    x = x.permute(1, 2, 0).numpy()
    y = y[0].cpu() 
    y = y.permute(1, 2, 0).numpy()
    z = z[0].cpu() 
    z = z.permute(1, 2, 0).numpy()
    w = w[0].cpu() 
    w = w.permute(1, 2, 0).numpy()
    v = v[0].cpu() 
    v = v.permute(1, 2, 0).numpy()

    f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
    x = x[:,:,::-1]
    y = y[:,:,::-1]
    z = z[:,:,::-1]
    w = w[:,:,::-1]
    v = v[:,:,::-1]
    
    ax1.imshow(x)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(y)
    ax2.set_title("Masked Image")
    ax2.axis('off')

    ax3.imshow(z)
    ax3.set_title("Reconstruction")
    ax3.axis('off')
    
    ax4.imshow(w)
    ax4.set_title("Inpainted GT")
    ax4.axis('off')

    ax5.imshow(v)
    ax5.set_title("Final Inpainted")
    ax5.axis('off')

    f.savefig('visualizations/inpaint_run.jpg')


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


def inpaint_loss(outputs, images):

    loss = nn.L1Loss()

    inputs = torch.stack(images,0).cuda()
    outputs = un_normalize(outputs)

    recon_loss =  loss(outputs, inputs)

    return recon_loss

def train(editor_model, inpaint_model, num_epochs, dataloader):

    optimizer = torch.optim.Adam(inpaint_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-2)

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

            originals, hole_images, _, inpainted_images = data

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.no_grad():
                reconstruction = editor_model(originals)

            reconstruction = un_normalize(reconstruction)
            reconstruction = torch.clamp(torch.round(reconstruction),min=0., max = 255.)

            originals = torch.stack(originals, dim=0).cuda()

            masks = torch.where((reconstruction-originals) < -30., 255., 0.)
            mask_s = torch.mean(masks,dim=1)
            masks = torch.where(mask_s > 80., 1., 0.) 
            mask_recons = torch.where(mask_s > 80., 255., 0.)

            masks = torch.stack([masks,masks,masks],dim=1)
            mask_recons = torch.stack([mask_recons,mask_recons,mask_recons],dim=1)

            reconstruction = reconstruction + mask_recons

            reconstruction = normalize(reconstruction)

            outputs = inpaint_model(reconstruction, masks)

            loss = inpaint_loss(outputs, inpainted_images)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            
            if i%30 == 0:
                print('Loss: {}'.format(loss.item()))
                reconstruction = un_normalize(reconstruction)
                outputs = un_normalize(outputs)
                inpainted_images = torch.stack(inpainted_images, dim=0)
                hole_inputs = torch.stack(hole_images, 0)
                visualize(originals*torch.tensor(1./255), hole_inputs*torch.tensor(1./255), reconstruction*torch.tensor(1./255), inpainted_images*torch.tensor(1./255), torch.round(outputs.detach())*torch.tensor(1./255))
            
            sys.stdout.flush()
        
        avg_loss = np.mean(running_loss)
        print("Epoch {}: Loss: {}".format(j+1,avg_loss))
        epoch_loss.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = j+1
            print('Model saved at Epoch: {}'.format(j+1))
            torch.save(inpaint_model.state_dict(),args.PATH)
    logger.info("Finished Training with best loss: {} at Epoch: {}".format(best_loss, best_epoch))
    plt.plot(np.linspace(1, num_epochs, num_epochs).astype(int), epoch_loss)
    if not os.path.exists('losses/'):
            os.makedirs('losses/')
    plt.savefig('losses/train_loss.png'.format(args.lr))


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
            loss = inpaint_loss(outputs, inputs)

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
    
    if args.eval:
        coco_test_loader, _ = get_loader(device=device, \
                                    root=args.coco+'val2017', \
                                        json=args.coco+'annotations/instances_val2017.json', \
                                            batch_size=args.batch_size, \
                                                shuffle=False, \
                                                    num_workers=0)
        inpainter_eval = Inpainter(64)
        inpainter_eval.load_state_dict(torch.load(args.PATH))
        eval(inpainter_eval.to(device),coco_test_loader)
    else:
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')

        editor = Editor(solo,reconstructor)
        editor.load_state_dict(torch.load(args.RECONS_PATH))
        editor.to(device)
        editor.eval()

        inpainter = Inpainter(64)
        inpainter.to(device)

        edit_params = sum(p.numel() for p in editor.parameters())
        inpaint_params = sum(p.numel() for p in inpainter.parameters())
    
        logger.info("Total Params: {}".format(edit_params+inpaint_params))
        logger.info("Trainable Params: {}".format(inpaint_params))
        logger.info("Non-Trainable Params: {}".format(edit_params))

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
            inpainter.load_state_dict(torch.load(args.PATH))
        
        train(editor_model=editor,inpaint_model=inpainter,num_epochs=args.num_epochs, dataloader=coco_train_loader)