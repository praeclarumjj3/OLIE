import torch.nn as nn
import torch.nn.functional as F
import torch

bench_maps = [(40,44), (50,58), (62,71), (74,83), (86,95), (98,107), (110,118), (112,130)]
car_maps = [(0,8), (12,20), (24,32), (36,44), (48,56), (60,63), (72,75), (84,87), (96,99), (108,111)]

def masking(image, phase, index):
    overlap = torch.ones_like(image)
    if phase=="single":
        for i in range(int(overlap.shape[0])):
            overlap[i][index] = torch.tensor(0, dtype = float)
    elif phase=="mult":
        for i in range(overlap.shape[0]):
            overlap[i][index[0]:index[1]] = torch.tensor(0, dtype = float)
    
    return image*overlap

def normalize(inputs):
    # pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).cuda().view(3, 1, 1)
    pixel_std = torch.Tensor([57.375, 57.120, 58.395]).view(3, 1, 1)
    normalizer = lambda x: (x) / pixel_std
    return normalizer(inputs)

class Reconstructor(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder()

    def forward(self, masks, images):
        images = normalize(images)
        images = F.interpolate(images,(64,64))
        masks = F.sigmoid(masks)
        masks = torch.ones_like(masks) - masks
        masks = torch.cat([masks,images], dim=1)
        x = self.encoder(masks)
        x = self.decoder(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels+3, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        return x
        

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1_1 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # dilated convolution blocks
        self.convA2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.convA2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        self.convA2_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8)
        self.convA2_4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=16, dilation=16)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1) 
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
        self.conv6 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2) 

    
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv1_3(x)
        x = F.upsample(x, scale_factor=2, mode='nearest')

        x = self.convA2_1(x)
        x = self.convA2_2(x)
        x = self.convA2_3(x)
        x = self.convA2_4(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv4a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv5(x)
        x = self.conv5a(x)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv6(x)

        return x