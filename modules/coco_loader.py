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
        self.root_bg = 'datasets/bg/'
        self.coco = COCO(json)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.device = device

    def __getitem__(self, index):
        """Returns one image."""
        coco = self.coco
        img_id = self.ids[index]
        path = coco.loadImgs(img_id)[0]['file_name']
        # ann_ids = []

        # for ann in self.coco.anns.keys():
        #     if coco.anns[ann]['image_id'] == img_id:
        #         ann_ids.append(ann)
        
        # b_boxes = []
        # for ann_id in ann_ids:
        #     b_boxes.append(coco.anns[ann_id]['bbox'])
        
        # for i in range(len(b_boxes)):
        #     for j in range(len(b_boxes[i])):
        #         b_boxes[i][j] = round(b_boxes[i][j])

        comp_img = read_image(os.path.join(self.root, path), format="BGR")
        comp_img = torch.from_numpy(comp_img.copy()).permute(2,0,1).float()

        bg = read_image(os.path.join(self.root_bg, path), format="BGR")
        bg = torch.from_numpy(bg.copy()).permute(2,0,1).float()

        # hole_image = torch.from_numpy(org.copy()).permute(2,0,1).float()
        # mask = torch.zeros_like(hole_image)
        # for b_box in b_boxes:
        #     hole_image[:,b_box[1]:b_box[1]+b_box[3],b_box[0]:b_box[0]+b_box[2]] = torch.Tensor([ 0., 0., 0.]).view(3, 1, 1)
        #     mask[:,b_box[1]:b_box[1]+b_box[3],b_box[0]:b_box[0]+b_box[2]] = torch.tensor(1.)

        comp_img = self.transform(comp_img)
        bg = self.transform(bg)
        # hole_image = self.transform(hole_image)
        # mask = self.transform(mask)

        if self.device is not None:
            return comp_img.to(self.device), bg.to(self.device)
        else:
            return comp_img, bg

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

    comp_imgs, bg_imgs = zip(*data)
    comp_imgs = list(comp_imgs)
    bg_imgs = list(bg_imgs)
    # hole_images = list(hole_images)
    # masks = list(masks)

    return comp_imgs, bg_imgs
    
def get_loader(device, root, json, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    
    transform = transforms.Compose([
        transforms.Resize((256,256))
    ])

    coco = CocoDataset(device=device,
                       root=root,
                       json=json,
                       transform=transform)
    
    coco= torch.utils.data.Subset(coco, list((range(0,int(len(coco)*0.03)))))
    
    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader, coco