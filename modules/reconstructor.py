import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.nn.modules import activation

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
        self.refiner = Refiner()

    def forward(self, masks, imgs):
        size = masks.shape[2]
        images = F.interpolate(imgs,(size,size))
        images = normalize(images)
        masks = F.tanh(masks)
        x = self.encoder(images,masks)
        x = self.decoder(x)
        x = self.refiner(x)

        return x

class GatedConv2dWithActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.batch_norm2d =nn.BatchNorm2d(out_channels)
   
    def forward(self, input):

        x = self.conv2d(input)
        mask = self.mask_conv2d(input)

        if self.activation is None:
            x = x * F.sigmoid(mask)
        else:
            x = self.activation(x) * F.sigmoid(mask)

        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x

class GatedDeConv2dWithActivation(nn.Module):
    
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        #print(input.size())
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 72, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(72)
        self.conv2 = nn.Conv2d(72, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)

        self.chan_conv = nn.Conv2d(144, 32, kernel_size=1, stride=1)

        self.cnum = 64

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.gated_conv1 = GatedConv2dWithActivation(in_channels=32, out_channels=self.cnum, kernel_size=5, padding=2)
        self.gated_conv1A = GatedConv2dWithActivation(self.cnum, 2*self.cnum, 3, 1, padding=1)
        # downsample
        
        self.gated_conv2 = GatedConv2dWithActivation(2*self.cnum, 2*self.cnum, 3, 1, padding=1)
        self.gated_conv2A = GatedConv2dWithActivation(2*self.cnum, 4*self.cnum, 3, 1, padding=1)
        #downsample

        self.gated_conv3 = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        self.gated_conv3A = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        
        # atrous convolution
        self.dil_conv4A = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, dilation=2, padding=2)
        self.dil_conv4B = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, dilation=4, padding=4)
        self.dil_conv4C = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, dilation=8, padding=8)
        self.dil_conv4D = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, dilation=16, padding=16)
        self.conv4E = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)

        self.conv5 = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        
    def forward(self, x, maps):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.4)

        ins = maps*x

        ins = self.chan_conv(ins)

        feats = self.gated_conv1(ins)
        feats = self.gated_conv1A(feats)
        feats = self.pool(feats)
        
        feats = self.gated_conv2(feats)
        feats = self.gated_conv2A(feats)
        feats = self.pool(feats)
        
        feats = self.gated_conv3(feats)
        feats = self.gated_conv3A(feats)

        feats = self.dil_conv4A(feats)
        feats = self.dil_conv4B(feats)
        feats = self.dil_conv4C(feats)
        feats = self.dil_conv4D(feats)
        feats = self.conv4E(feats)

        feats = self.conv5(feats)

        return feats

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.cnum = 64

        # upsample
        self.conv1 = GatedDeConv2dWithActivation(2, 4*self.cnum, 2*self.cnum, 3, 1, padding=1)
        self.conv1A = GatedConv2dWithActivation(2*self.cnum, 2*self.cnum, 3, 1, padding=1)
        
        #upsample
        self.conv2 = GatedDeConv2dWithActivation(2, 2*self.cnum, self.cnum, 3, 1, padding=1)
        self.conv2A = GatedConv2dWithActivation(self.cnum, self.cnum//2, 3, 1, padding=1)
        
        self.conv3 = GatedConv2dWithActivation(self.cnum//2, 3, 3, 1, padding=1, activation=None)


    def forward(self, feats):
       
        feats = self.conv1(feats)
        feats = self.conv1A(feats)

        feats = F.upsample(input=feats, scale_factor=2, mode='bilinear')

        feats = self.conv2(feats)
        feats = self.conv2A(feats)

        feats = F.upsample(input=feats, scale_factor=2, mode='bilinear')

        feats = self.conv3(feats)
        
        return feats


class Refiner(nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        self.encoder = RefineEncoder()
        self.decoder = RefineDecoder()

    def forward(self, reconstruction):

        x = self.encoder(reconstruction)
        x = self.decoder(x)

        return x


class RefineEncoder(nn.Module):
    def __init__(self):
        super(RefineEncoder, self).__init__()

        self.cnum = 32

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.gated_conv1 = GatedConv2dWithActivation(in_channels=3, out_channels=self.cnum, kernel_size=5, padding=2)
        self.gated_conv1A = GatedConv2dWithActivation(self.cnum, 2*self.cnum, 3, 1, padding=1)
        # downsample
        
        self.gated_conv2 = GatedConv2dWithActivation(2*self.cnum, 2*self.cnum, 3, 1, padding=1)
        self.gated_conv2A = GatedConv2dWithActivation(2*self.cnum, 4*self.cnum, 3, 1, padding=1)
        #downsample

        self.gated_conv3 = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        self.gated_conv3A = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        self.gated_conv3B = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        
        # atrous convolution
        self.dil_conv4A = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, dilation=2, padding=2)
        self.dil_conv4B = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, dilation=4, padding=4)
        self.dil_conv4C = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, dilation=8, padding=8)
        self.dil_conv4D = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, dilation=16, padding=16)
        self.conv4E = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
    
    def forward(self, feats):

        # refine-encoder
        feats = self.gated_conv1(feats)
        feats = self.gated_conv1A(feats)
        feats = self.pool(feats)
        
        feats = self.gated_conv2(feats)
        feats = self.gated_conv2A(feats)
        feats = self.pool(feats)
        
        feats = self.gated_conv3(feats)
        feats = self.gated_conv3A(feats)
        feats = self.gated_conv3B(feats)

        feats = self.dil_conv4A(feats)
        feats = self.dil_conv4B(feats)
        feats = self.dil_conv4C(feats)
        feats = self.dil_conv4D(feats)
        feats = self.conv4E(feats)

        return feats

class RefineDecoder(nn.Module):
    def __init__(self):
        super(RefineDecoder, self).__init__()

        self.cnum = 64

        # upsample
        self.conv1 = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        self.conv1A = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        self.conv1B = GatedDeConv2dWithActivation(2, 4*self.cnum, 2*self.cnum, 3, 1, padding=1)
        self.conv1C = GatedConv2dWithActivation(2*self.cnum, 2*self.cnum, 3, 1, padding=1)
        
        #upsample
        self.conv2 = GatedDeConv2dWithActivation(2, 2*self.cnum, self.cnum, 3, 1, padding=1)
        self.conv2A = GatedConv2dWithActivation(self.cnum, self.cnum//2, 3, 1, padding=1)
        
        self.conv3 = GatedConv2dWithActivation(self.cnum//2, 3, 3, 1, padding=1, activation=None)
    
    def forward(self, feats):

        # refine-decoder
        feats = self.conv1(feats)
        feats = self.conv1A(feats)
        feats = self.conv1B(feats)
        feats = self.conv1C(feats)

        feats = self.conv2(feats)
        feats = self.conv2A(feats)

        feats = self.conv3(feats)
        
        return feats