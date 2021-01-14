import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from pycocotools.coco import COCO
from detectron2.data.detection_utils import read_image

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
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        """Returns one image."""
        coco = self.coco
        ann_id = self.ids[index]
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']

        image = read_image(os.path.join(self.root, path), format="BGR")
        image = torch.from_numpy(image.copy()).permute(2,0,1).float()
        image = self.transform(image)
        # image = self.transform(image)
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
        transforms.CenterCrop(256),
        transforms.Resize((256,256))
    ])

    coco = CocoDataset(device=device,
                       root=root,
                       json=json,
                       transform=transform)
    
    coco= torch.utils.data.Subset(coco, list((range(0,int(len(coco)*0.02)))))
    
    # Data loader for COCO dataset
    # This will return (images)
    # images: a tensor of shape (batch_size, 3, 224, 224).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader, coco