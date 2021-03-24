import torch
from adet.config import get_cfg
from modules.solov2 import SOLOv2
import argparse
import os
import warnings
from detectron2.utils.logger import setup_logger
import glob
import time
import torchvision.transforms as transforms
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.detection_utils import read_image
from PIL import Image
from etaprogress.progress import ProgressBar
import requests

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
    return parser

def inpaint(solo, args):
    
    transform = transforms.Compose([
        transforms.Resize((300,300))
    ])
    
    if os.path.isdir(args.input[0]):
        args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    
    total = len(args.input)
    bar = ProgressBar(total, max_width=80)
    for i, path in enumerate(args.input):
        bar.numerator = i+1
        print(bar, end='\r')
        image = read_image(path, format="BGR")
        img = torch.from_numpy(image.copy()).permute(2,0,1).float()
        img = transform(img)

        batched_input = []
        batched_input.append(img)
        
        mask, _ = solo(batched_input)
        
        mask = (mask[0] > 0).float().cpu()
        img = img.cpu()

        image = img.permute(1, 2, 0).numpy()
        mask = mask.squeeze(0).numpy() * 255.

        image = image[:,:,::-1]

        image = image.astype('uint8')
        mask = mask.astype('uint8')

        image = Image.fromarray(image,'RGB')
        mask = Image.fromarray(mask,'L')
        
        mode_img = image.mode
        mode_msk = mask.mode

        W, H = image.size
        str_img = image.tobytes().decode("latin1")
        str_msk = mask.tobytes().decode("latin1")

        data = {'str_img': str_img, 'str_msk': str_msk, 'width':W, 'height':H, 
                'mode_img':mode_img, 'mode_msk':mode_msk,  'is_refine': True}
        
        time.sleep(0.01)
        r = requests.post('http://47.57.135.203:2333/api', json=data)
        str_result = r.json()['str_result']
        result = str_result.encode("latin1")
        result = Image.frombytes('RGB', (W, H), result, 'raw')

        images = [image, mask, result]
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]

        new_im.save("baselines/profill_on_solo/results/{}.jpg".format(i))
    print('\n')
               

if __name__ == "__main__":
    logger = setup_logger()
    args = get_parser().parse_args()
    
    logger.info("Arguments: " + str(args))

    if not os.path.exists('baselines/profill_on_solo/results/'):
        os.makedirs('baselines/profill_on_solo/results/')

    cfg = setup_cfg(args)

    solo = SOLOv2(cfg=cfg)
    checkpointer = DetectionCheckpointer(solo)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    for param in solo.parameters():
        param.requires_grad = False
    
    inpaint(solo=solo.eval(), args=args)