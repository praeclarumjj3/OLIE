import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
        img = coco.loadImgs(img_id)[0]
        path = img['file_name']
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        mask = np.zeros((img['height'],img['width']))

        for i in range(len(anns)):
            if np.mean(mask) > 0.2:
                break
            mask = np.maximum(coco.annToMask(anns[i]), mask)
        
        org = Image.open(os.path.join(self.root, path))
        
        if org.mode != 'RGB':
            org = org.convert('RGB')
        
        org = np.asarray(org, dtype="float64")
        
        mask = np.expand_dims(mask, axis=2)

        image = torch.from_numpy(org.copy()).permute(2,0,1).float()
        mask = torch.from_numpy(mask.copy()).permute(2,0,1).float()

        image = self.transform(image)
        mask = self.transform(mask)
    
        return image, mask, path

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

    images, masks, paths = zip(*data)
    images = list(images)
    masks = list(masks)
    paths = list(paths)

    return images, masks, paths
    
def get_loader(root, json, shuffle):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    
    transform = transforms.Compose([
        transforms.Resize((300,300))
    ])

    coco = CocoDataset(root=root,
                       json=json,
                       transform=transform)
    
    coco= torch.utils.data.Subset(coco, list((range(0,5))))
    
    # Data loader for COCO dataset
    # This will return (images)
    # images: a tensor of shape (batch_size, 3, 224, 224).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=1,
                                              shuffle=shuffle,
                                              num_workers=0,
                                              collate_fn=collate_fn)
    return data_loader