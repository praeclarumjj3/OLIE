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
    def __init__(self, device, root, json, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            transform: image transformer.
        """
        self.root = root
        self.root_in = self.root + '_inpainted'
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.device = device

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
            mask = np.maximum(coco.annToMask(anns[i]), mask)
        
        org = Image.open(os.path.join(self.root, path))
        org = np.asarray(org, dtype="float64")
        
        mask = np.expand_dims(mask, axis=2)
        
        org = org*mask
        
        # plt.imsave('visualizations/mask.jpg',np.squeeze(mask))
        # plt.imsave('visualizations/org.jpg',org.astype(np.uint8))
        # exit()

        image = torch.from_numpy(org.copy()).permute(2,0,1).float()
        mask = torch.from_numpy(mask.copy()).permute(2,0,1).float()

        image = self.transform(image)
        mask = self.transform(mask)
    
        if self.device is not None:
            return image.to(self.device), mask.to(self.device)
        else:
            return image, mask

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

    images, masks = zip(*data)
    images = list(images)
    masks = list(masks)

    return images, masks
    
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
    
    coco= torch.utils.data.Subset(coco, list((range(0,int(len(coco)*0.015)))))
    
    # Data loader for COCO dataset
    # This will return (images)
    # images: a tensor of shape (batch_size, 3, 224, 224).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader, coco