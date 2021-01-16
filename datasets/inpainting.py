from datasets.gen_masked_image_loader import get_loader
import torch
from torch import nn

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # coco_train_loader = get_loader(device=device, \
    #                                 root='datasets/coco/train2017', \
    #                                     json='datasets/coco/annotations/instances_train2017.json', \
    #                                         batch_size=16, \
    #                                             shuffle=True, \
    #                                                 num_workers=0)

    coco_test_loader = get_loader(device=device, \
                                    root='datasets/coco/val2017', \
                                        json='datasets/coco/annotations/instances_val2017.json', \
                                            batch_size=16, \
                                                shuffle=True, \
                                                    num_workers=0)

    
    for i, data in enumerate(coco_test_loader, 0):
        inputs = data
        exit()
