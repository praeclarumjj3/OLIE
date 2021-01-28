# -*- coding: utf-8 -*-
from torch._C import device
import torch
from torch import nn
from adet.config import get_cfg
from modules.solov2 import SOLOv2
from modules.reconstructor import Reconstructor
import matplotlib.pyplot as plt
import argparse
import os
import warnings
from detectron2.utils.logger import setup_logger
import glob
import time
from PIL import Image
import torchvision.transforms as transforms
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from run import Editor


warnings.filterwarnings("ignore")

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
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--PATH",
        help="Path of the saved editor",
        default='checkpoints/editor.pth',
        type=str
    )
    return parser


def un_normalize(inputs):
    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor([1.0, 1.0, 1.0]).to(device).view(3, 1, 1)
    un_normalizer = lambda x: (x + pixel_mean) * pixel_std
    return un_normalizer(inputs)


def demo(editor, args):
    
    transform = transforms.Compose([
        transforms.Resize((640,640))
    ])
    
#     editor_demo2 =Editor(solo,reconstructor)
#     editor_demo2.load_state_dict(torch.load('checkpoints/editor_loss_complete.pth'))
#     editor_demo2.cuda()
    
    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for i, path in enumerate(args.input):
        image = read_image(path, format="BGR")
        img = torch.from_numpy(image.copy()).permute(2,0,1).float()
        img = transform(img)
        
        hole_image = read_image('inputs/hole_image_train{}.jpg'.format(i+1), format="BGR")
        hole_img = torch.from_numpy(hole_image.copy()).permute(2,0,1).float()
        hole_img = transform(hole_img)
        
        f, (ax1,ax2,ax3) = plt.subplots(1,3)
        org = img * torch.tensor(1./255)
        org = org.cpu().permute(1,2,0).numpy()
        org = org[:,:,::-1]
        ax1.imshow(org)
        ax1.set_title("Image")
        ax1.axis('off')
        batched_input = []
        batched_input.append(img)
        logger.info("Starting Visualization")
        start_time = time.time()
        with torch.no_grad():
            reconstruction_1 = editor(batched_input, [hole_img])
#             reconstruction_2 = editor(batched_input)
        end_time = time.time()
        logger.info("Duration: {}".format(end_time-start_time))
        reconstruction_1 = un_normalize(reconstruction_1)
#         reconstruction_2 = un_normalize(reconstruction_2)
        reconstruction_1 = torch.clamp(torch.round(reconstruction_1.squeeze(0).cpu()),min=0., max = 255.) * torch.tensor(1./255)
        reconstruction_1 = reconstruction_1.permute(1, 2, 0).numpy()
        reconstruction_1 = reconstruction_1[:,:,::-1]
        
#         reconstruction_2 = torch.clamp(torch.round(reconstruction_2.squeeze(0).cpu()),min=0., max = 255.) * torch.tensor(1./255)
#         reconstruction_2 = reconstruction_2.permute(1, 2, 0).numpy()
#         reconstruction_2 = reconstruction_2[:,:,::-1]
        hole_org = hole_img * torch.tensor(1./255)
        hole_org = hole_org.cpu().permute(1,2,0).numpy()
        hole_org = hole_org[:,:,::-1]
        ax2.imshow(hole_org)
        ax2.set_title("Hole Image")
        ax2.axis('off')

        ax3.imshow(reconstruction_1)
        ax3.set_title("Reconstruction")
        ax3.axis('off')
#         ax4.imshow(reconstruction_2)
#         ax4.set_title("Complete Loss")
#         ax4.axis('off')
        f.savefig('visualizations/train_demo{}.jpg'.format(i+1))

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

    if not os.path.exists('visualizations/'):
        os.makedirs('visualizations/')
        logger.info("Instantiating Editor")
    editor_demo =Editor(solo,reconstructor)
    editor_demo.load_state_dict(torch.load(args.PATH))
    demo(editor=editor_demo.cuda(), args=args)
