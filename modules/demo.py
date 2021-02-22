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
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(device).view(3, 1, 1).cuda()
    pixel_std = torch.Tensor([57.375, 57.120, 58.395]).view(3, 1, 1).cuda()
    un_normalizer = lambda x: x * pixel_std + pixel_mean
    return un_normalizer(inputs)

def demo_replacement(editor, args):
    
    transform = transforms.Compose([
        transforms.Resize((640,640))
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
        
        f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4))
        org = img * torch.tensor(1./255)
        org = org.cpu().permute(1,2,0).numpy()
        org = org[:,:,::-1]

        plt.rcParams.update({'font.size': 16})
        ax1.imshow(org)
        ax1.set_title("Image")
        ax1.axis('off')

        batched_input = []
        batched_input.append(img)

        logger.info("Starting Visualization")
        start_time = time.time()
        
        with torch.no_grad():
           reconstruction = editor(batched_input)     

        reconstruction = un_normalize(reconstruction)
        reconstruction = torch.clamp(torch.round(reconstruction.squeeze(0).cpu()),min=0., max = 255.)

        masked_image = reconstruction
        masked_image = img - masked_image
        masked_image = torch.clamp(torch.round(masked_image.cpu()),min=0., max = 255.) * torch.tensor(1./255)
        masked_image = masked_image.permute(1, 2, 0).numpy()
        masked_image = masked_image[:,:,::-1]

        reconstruction = reconstruction * torch.tensor(1./255)
        reconstruction = reconstruction.permute(1, 2, 0).numpy()
        reconstruction = reconstruction[:,:,::-1]

        end_time = time.time()
        logger.info("Duration: {}".format(end_time-start_time))
        
        plt.rcParams.update({'font.size': 16})
        ax2.imshow(reconstruction)
        ax2.set_title("Reconstruction")
        ax2.axis('off')

        plt.rcParams.update({'font.size': 16})
        ax3.imshow(masked_image)
        ax3.set_title("Image - Reconstruction")
        ax3.axis('off')

        f.savefig('visualizations/val_demo{}.jpg'.format(i+1))

def tensor_to_list(maps):

    masks = []
    maps = maps.squeeze(0)

    for i in range(maps.shape[0]):
        masks.append(maps[i])

    return masks

def demo_replacement(editor, args):
    
    transform = transforms.Compose([
        transforms.Resize((640,640))
    ])
    
    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for i, path in enumerate(args.input):
        image = read_image(args.input[0], format="BGR")
        image1 = read_image(args.input[1], format="BGR")

        img = torch.from_numpy(image.copy()).permute(2,0,1).float()
        img = transform(img)

        img1 = torch.from_numpy(image1.copy()).permute(2,0,1).float()
        img1 = transform(img1)
        
        f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(12,4))
        org = img * torch.tensor(1./255)
        org = org.cpu().permute(1,2,0).numpy()
        org = org[:,:,::-1]

        f1, (ax11,ax21,ax31) = plt.subplots(1,3,figsize=(12,4))
        org1 = img1 * torch.tensor(1./255)
        org1 = org1.cpu().permute(1,2,0).numpy()
        org1 = org1[:,:,::-1]

        plt.rcParams.update({'font.size': 16})
        ax1.imshow(org)
        ax1.set_title("Image")
        ax1.axis('off')

        plt.rcParams.update({'font.size': 16})
        ax11.imshow(org1)
        ax11.set_title("Image")
        ax11.axis('off')

        batched_input = []
        batched_input.append(img)

        batched_input1 = []
        batched_input1.append(img1)

        logger.info("Starting Visualization")
        start_time = time.time()
        
        with torch.no_grad():
            masks, ins = editor.solo(batched_input)
            masks1, ins1 = editor.solo(batched_input1)

            maps = tensor_to_list(masks)
            maps1 = tensor_to_list(masks1)

            maps_in = maps[0:72] + maps1[72:144]
            maps_in1 = maps1[0:72] + maps[72:144]

            maps_in = torch.stack(maps_in,dim=0)
            maps_in1 = torch.stack(maps_in1,dim=0)

            maps_in = maps_in.unsqueeze(0)
            maps_in1 = maps_in1.unsqueeze(0)

            reconstruction = editor.reconstructor(maps_in, ins)
            reconstruction1 = editor.reconstructor(maps_in1, ins1)            

        reconstruction = un_normalize(reconstruction)
        reconstruction = torch.clamp(torch.round(reconstruction.squeeze(0).cpu()),min=0., max = 255.)

        reconstruction1 = un_normalize(reconstruction1)
        reconstruction1 = torch.clamp(torch.round(reconstruction1.squeeze(0).cpu()),min=0., max = 255.)

        masked_image = reconstruction
        masked_image = img - masked_image
        masked_image = torch.clamp(torch.round(masked_image.cpu()),min=0., max = 255.) * torch.tensor(1./255)
        masked_image = masked_image.permute(1, 2, 0).numpy()
        masked_image = masked_image[:,:,::-1]

        masked_image1 = reconstruction1
        masked_image1 = img1 - masked_image1
        masked_image1 = torch.clamp(torch.round(masked_image1.cpu()),min=0., max = 255.) * torch.tensor(1./255)
        masked_image1 = masked_image1.permute(1, 2, 0).numpy()
        masked_image1 = masked_image1[:,:,::-1]

        reconstruction = reconstruction * torch.tensor(1./255)
        reconstruction = reconstruction.permute(1, 2, 0).numpy()
        reconstruction = reconstruction[:,:,::-1]

        reconstruction1 = reconstruction1 * torch.tensor(1./255)
        reconstruction1 = reconstruction1.permute(1, 2, 0).numpy()
        reconstruction1 = reconstruction1[:,:,::-1]
        
        end_time = time.time()
        logger.info("Duration: {}".format(end_time-start_time))
        
        plt.rcParams.update({'font.size': 16})
        ax2.imshow(reconstruction)
        ax2.set_title("Reconstruction")
        ax2.axis('off')

        plt.rcParams.update({'font.size': 16})
        ax21.imshow(reconstruction1)
        ax21.set_title("Reconstruction")
        ax21.axis('off')

        plt.rcParams.update({'font.size': 16})
        ax3.imshow(masked_image)
        ax3.set_title("Image - Reconstruction")
        ax3.axis('off')

        plt.rcParams.update({'font.size': 16})
        ax31.imshow(masked_image1)
        ax31.set_title("Image - Reconstruction")
        ax31.axis('off')

        f.savefig('visualizations/val_demo{}.jpg'.format(i+1))
        f1.savefig('visualizations/val_demo{}.jpg'.format(i+2))

        exit()

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
    editor_demo =Editor(solo, reconstructor)
    editor_demo.load_state_dict(torch.load(args.PATH))
    demo(editor=editor_demo.cuda(), args=args)