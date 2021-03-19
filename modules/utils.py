import torch
import matplotlib.pyplot as plt
import random
from torchvision import utils
import numpy as np

def masking(image, phase, index):
    if phase=="single":
        # print(torch.mean(image[i][index]))
        for i in range(int(image.shape[0])):
            image[i][index] = torch.tensor(0, dtype = float)
    elif phase=="multi":
        for i in range(image.shape[0]):
            # print(torch.mean(image[i][index[0]:index[1]]))
            image[i][index[0]:index[1]] = torch.tensor(0, dtype = float)
    
    return image

def masking_objects(image, phase, c_index, x, w, y, h):
    if phase=="single":
        # print(torch.mean(image[i][c_index,y:y+h,x:x+w]))
        for i in range(int(image.shape[0])):
            image[i][c_index,y:y+h,x:x+w] = torch.tensor(0, dtype = float)
    elif phase=="multi":
        for i in range(image.shape[0]):
            # print(torch.mean(image[i][c_index[0]:c_index[1],y:y+h,x:x+w]))
            image[i][c_index[0]:c_index[1],y:y+h,x:x+w] = torch.tensor(0, dtype = float)
    
    return image

def masking_threshold(threshold, image, phase, c_index, x, w, y, h):
    overlap = torch.ones_like(image)
    if phase=="single":
        for i in range(int(overlap.shape[0])):
            # print(torch.mean(image[i][c_index,y:y+h,x:x+w]))
            overlap[i][c_index,y:y+h,x:x+w] = (image[i][c_index,y:y+h,x:x+w] > threshold).float() 
    elif phase=="multi":
        for i in range(overlap.shape[0]):
            # print(torch.mean(image[i][c_index[0]:c_index[1],y:y+h,x:x+w]))
            overlap[i][c_index[0]:c_index[1],y:y+h,x:x+w] = (image[i][c_index[0]:c_index[1],y:y+h,x:x+w] > threshold).float()
    
    return image*overlap

def normalize(inputs):
    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(3, 1, 1).cuda()
    pixel_std = torch.Tensor([57.375, 57.120, 58.395]).view(3, 1, 1).cuda()
    normalizer = lambda x: (x.cuda() - pixel_mean) / pixel_std
    return normalizer(inputs)

def mask_shuffle(image, index, phase):

    masks_start = []
    masks = []
    image = image.squeeze(0)

    for i in range(index[0]):
        masks_start.append(image[i])

    for i in range(index[0],index[1]):
        masks.append(image[i])
    
    if phase == "reverse":
        masks.reverse()
    elif phase == "random":
        random.shuffle(masks)

    masks = masks_start + masks

    for i in range(index[1],image.shape[0]):
        masks.append(image[i])

    masks = torch.stack(masks, dim=0)

    return masks.unsqueeze(0)

def visualize(x,layer):
    plt.rcParams.update({'font.size': 3})
    dim = int(x.shape[1])
    x = x[0].cpu() 
    x = x.permute(1, 2, 0).numpy()
    f, axarr = plt.subplots(int(dim**0.5),int(dim**0.5),figsize=(16,16))
    for j in range(int(dim**0.5)*int(dim**0.5)):
        r = int(j/int(dim**0.5))
        c = int(j%int(dim**0.5))
        axarr[r,c].imshow(x[:,:,j])
        axarr[r,c].axis('off')
    f.savefig('visualizations/{}.jpg'.format(layer))


def visTensor(tensor, name, ch=0, allkernels=False, nrow=8, padding=1): 
    
    n,c,w,h = tensor.shape

    if allkernels: 
        tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: 
        tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    
    plt.figure(figsize=(nrow,rows))
    plt.axis('off')
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
    plt.savefig('visualizations/{}.jpg'.format(name))
    
def visualize_kernels(model, name):
    if len(name) == 0:
        return

    filters = getattr(model,name)
    filters = filters.weight.data.clone()

    visTensor(filters, name, ch=0, allkernels=False)
