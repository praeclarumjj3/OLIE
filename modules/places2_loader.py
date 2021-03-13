import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def get_places2_loader(root, batch_size, isTrain=True):
    if isTrain:
        dir = os.path.join(root, 'train')
    else:
        dir = os.path.join(root.places2, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolderWithPaths(
        root = dir,
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            # normalize,
        ]))
    
    dataset = torch.utils.data.Subset(dataset, list((range(0,int(len(dataset)*0.015)))))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0)
    
    return loader