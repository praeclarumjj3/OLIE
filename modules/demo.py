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
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(device).view(3, 1, 1)
    un_normalizer = lambda x: (x + pixel_mean) * pixel_std
    return un_normalizer(inputs)


def recons_loss(outputs, images):
    loss = nn.L1Loss()
    inputs = torch.stack(images,0).cuda()
    outputs = un_normalize(outputs)
    return loss(inputs ,outputs)


def demo(editor, args):
    
    transform = transforms.Compose([
        transforms.CenterCrop(256),
        transforms.Resize((256,256))
    ])
    
    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for i, path in enumerate(args.input):
        image = read_image(path, format="BGR")
        img = torch.from_numpy(image.copy()).permute(2,0,1).float()
        img = transform(img)
        f, (ax1,ax2) = plt.subplots(1,2)
        org = img * torch.tensor(1./255)
        org = org.cpu().permute(1,2,0).numpy()
        org = org[:,:,::-1]
        ax1.imshow(org)
        batched_input = []
        batched_input.append(img)
        logger.info("Starting Visualization")
        start_time = time.time()
        with torch.no_grad():
            reconstruction = editor(batched_input)
        end_time = time.time()
        logger.info("Duration: {}".format(end_time-start_time))
        reconstruction = torch.clamp(torch.round(reconstruction.squeeze(0).cpu()),min=0., max = 255.) * torch.tensor(1./255)
        reconstruction = reconstruction.permute(1, 2, 0).numpy()
        reconstruction = reconstruction[:,:,::-1]
        ax2.imshow(reconstruction)
        f.savefig('visualizations/demo{}.jpg'.format(i))

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
