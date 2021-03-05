import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import random
from gated_modules import Refiner, GatedEncoder, GatedDecoder 

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

class Reconstructor(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.encoder = Encoder(in_channels)
        self.decoder = Decoder()

    def forward(self, masks, imgs):
        size = masks.shape[2]
        images = F.interpolate(imgs,(size,size))
        images = normalize(images)
        masks = F.tanh(masks)
        x = self.encoder(images,masks)
        x = self.decoder(x)

        return x

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 72, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(72)
        self.conv2 = nn.Conv2d(72, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.chan_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        
        self.cnum = 64

        self.enc_conv1 = nn.Conv2d(32,self.cnum,kernel_size=3,stride=1,padding=1)
        self.enc_conv1A = nn.Conv2d(self.cnum, 2*self.cnum, 3, 1, padding=1)

        self.enc_conv2 = nn.Conv2d(2*self.cnum, 2*self.cnum, 3, 1, padding=1)
        self.enc_conv2A = nn.Conv2d(2*self.cnum, 4*self.cnum, 3, 1, padding=1)

        self.enc_conv3 = nn.Conv2d(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        self.enc_conv3A = nn.Conv2d(4*self.cnum, 4*self.cnum, 3, 1, padding=1)

        self.conv5 = nn.Conv2d(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        
    def forward(self, x, maps):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.4)

        ins = maps*x

        ins = self.chan_conv(ins)

        feats = self.enc_conv1(ins)
        feats = F.leaky_relu(feats, negative_slope=0.4)
        feats = self.enc_conv1A(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
        
        feats = self.enc_conv2(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
        feats = self.enc_conv2A(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
        
        feats = self.enc_conv3(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
        feats = self.enc_conv3A(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)

        feats = self.conv5(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)

        return feats

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.cnum = 64

        # upsample
        self.conv1 = nn.Conv2d(4*self.cnum, 2*self.cnum, 3, 1, padding=1)
        self.conv1A = nn.Conv2d(2*self.cnum, 2*self.cnum, 3, 1, padding=1)
        
        #upsample
        self.conv2 = nn.Conv2d(2*self.cnum, self.cnum, 3, 1, padding=1)
        self.conv2A = nn.Conv2d(self.cnum, self.cnum//2, 3, 1, padding=1)
        
        self.conv3 = nn.Conv2d(self.cnum//2, 3, 3, 1, padding=1)


    def forward(self, feats):
       
        feats = self.conv1(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
        feats = F.interpolate(feats, scale_factor=2)
        feats = self.conv1A(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)

        feats = self.conv2(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
        feats = F.interpolate(feats, scale_factor=2)
        feats = self.conv2A(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)

        feats = self.conv3(feats)
        
        return feats