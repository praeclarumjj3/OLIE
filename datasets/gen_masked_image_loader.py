import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, device, root, json, transform=None):
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
        self.device = device

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

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        f, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(image)
        image = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()
        for b_box in b_boxes:
            image[:,b_box[1]:b_box[1]+b_box[3],b_box[0]:b_box[0]+b_box[2]] = torch.tensor(1.)
        image = image * torch.tensor(1./255)
        
        image = self.transform(image)

        if len(b_boxes) == 0:
            x = round(image.shape[2] / 2)
            y = round(image.shape[1] / 2)
            image[:,y-32:y+32,x-32:x+32] = torch.tensor(0.)
        
        # image = image.permute(1, 2, 0).numpy()
        # ax2.imshow(image)
        # f.savefig('masked_image.jpg')
        # exit()

        image = self.transform(image)
        if self.device is not None:
            return image.to(self.device)
        else:
            return image

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

    images = data
    images = list(images)

    return images

def get_loader(device, root, json, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    
    transform = transforms.Compose([
        transforms.Resize((640,640))
    ])
    
    coco = CocoDataset(device=device,
                       root=root,
                       json=json,
                       transform=transform)
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader