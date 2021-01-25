import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image
from matplotlib import cm
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import json


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __getitem__(self, index):
        """Returns one image."""
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        ann_ids = []
        for ann in self.coco.anns.keys():
            if coco.anns[ann]['image_id'] == img_id:
                ann_ids.append(ann)
        
        b_boxes = []
        for ann_id in ann_ids:
            b_boxes.append(coco.anns[ann_id]['bbox'])
        
        for i in range(len(b_boxes)):
            for j in range(len(b_boxes[i])):
                b_boxes[i][j] = round(b_boxes[i][j])

        org = read_image(os.path.join(self.root, path), format="BGR")
        hole_image = torch.from_numpy(image.copy()).permute(2,0,1).float()
        mask = torch.zeros_like(hole_image)
        for b_box in b_boxes:
            hole_image[:,b_box[1]:b_box[1]+b_box[3],b_box[0]:b_box[0]+b_box[2]] = torch.tensor(255.)
            mask[:,b_box[1]:b_box[1]+b_box[3],b_box[0]:b_box[0]+b_box[2]] = torch.tensor(255.)
        # image = image * torch.tensor(1./255)

        # hole_image = hole_image * torch.tensor(1./255)
        # mask = mask * torch.tensor(1./255)
        
        # image = self.transform(image)
        hole_image = self.transform(hole_image)
        mask = self.transform(mask)

        if len(b_boxes) == 0:
            x = round(hole_image.shape[2] / 2)
            y = round(hole_image.shape[1] / 2)
            hole_image[:,y-32:y+32,x-32:x+32] = torch.tensor(255.)
            mask[:,y-32:y+32,x-32:x+32] = torch.tensor(255.)
        
        # image = image.permute(1, 2, 0).numpy()
        hole_image = hole_image.permute(1, 2, 0).numpy()
        mask = mask.permute(1, 2, 0).numpy()
        
        hole_image = hole_image.astype('uint8')
        mask = mask.astype('uint8')
        
        hole_image = Image.fromarray(hole_image,'RGB')
        mask = Image.fromarray(mask,'RGB')

        # plt.imsave("org.jpg", image)
        # mask.save("mask.png")
        # hole_image.save("hole_image.jpg")
        # ax2.imshow(image)
        # ax3.imshow(mask)
        # f.savefig('masked_image.jpg')
        # exit()
        
        return hole_image, mask, path

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of images.
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list
            - image: torch tensor of shape (3, 256, 256).
            
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        
    """

    hole_images, masks, paths = zip(*data)
    hole_images = list(hole_images)
    masks = list(masks)
    paths = list(paths)

    return hole_images, masks, paths

def get_loader(root, json, batch_size, shuffle):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    
    transform = transforms.Compose([
        # transforms.CenterCrop(256),
        transforms.Resize((256,256))
    ])
    
    coco = CocoDataset(root=root,
                       json=json,
                       transform=transform)

    coco= torch.utils.data.Subset(coco, list((range(0,int(len(coco)*0.018)))))

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader