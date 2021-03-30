import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch


def get_places2_loader(root, batch_size):
    dir = os.path.join(root)

    dataset = datasets.ImageFolder(
        root = dir,
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            # normalize,
        ]))
    
    dataset = torch.utils.data.Subset(dataset, list((range(0,118287))))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0)
    
    return loader