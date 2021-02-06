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
    pixel_mean = torch.Tensor([103.530, 116.280, 123.675]).view(3, 1, 1).cuda()
    pixel_std = torch.Tensor([57.375, 57.120, 58.395]).view(3, 1, 1).cuda()
    normalizer = lambda x: (x.cuda() - pixel_mean) / pixel_std
    return normalizer(inputs)

class Reconstructor(nn.Module):

    def __init__(self, encoder, decoder, base_decoder):
        super().__init__()

        self.encoder = encoder
        self.base_decoder = base_decoder
        self.decoder = decoder

    def forward(self, masks, images):
        size = masks.shape[2]
        images = F.interpolate(images,(size,size))
        images = normalize(images)
        masks = F.sigmoid(masks)
        masks = torch.ones_like(masks) - masks
        masks = torch.cat([masks,images], dim=1)
        x = self.encoder(masks)
        x = self.base_decoder(x)
        x = self.decoder(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels+3, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv2(x)
        x = self.bn2(x)

        return x

class BaseDecoder(nn.Module):
    def __init__(self):
        super(BaseDecoder, self).__init__()
        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        # dilated convolution blocks
        self.convA2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.convA2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        self.convA2_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1) 
        # self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        # self.conv5a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
        # self.conv6 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2) 

    
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv1_2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.4)

        x = self.convA2_1(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.convA2_2(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.convA2_3(x)
        x = F.leaky_relu(x, negative_slope=0.4)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv4(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv4a(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = F.upsample(x, scale_factor=2, mode='nearest')
        # x = self.conv5(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv5a(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = F.upsample(x, scale_factor=2, mode='nearest')
        # x = self.conv6(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        
        return x        

class OrigDecoder(nn.Module):
    def __init__(self):
        super(OrigDecoder, self).__init__()
        # self.conv1_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # # dilated convolution blocks
        # self.convA2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        # self.convA2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.convA2_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8)

        # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1) 
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
        self.conv6 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2) 

    
    def forward(self, x):
        # x = self.conv1_1(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv1_2(x)
        # x = F.leaky_relu(x, negative_slope=0.4)

        # x = self.convA2_1(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.convA2_2(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.convA2_3(x)
        # x = F.leaky_relu(x, negative_slope=0.4)

        # x = self.conv3(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv4a(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv5(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv5a(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv6(x)
        
        return x

class EditDecoder(nn.Module):
    def __init__(self):
        super(EditDecoder, self).__init__()
        # self.conv1_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        # self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        # # dilated convolution blocks
        # self.convA2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        # self.convA2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.convA2_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8)

        # self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.perb4 = Perturbor(shape=(4,128,160,160)) 
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.perb4a = Perturbor(shape=(4,128,160,160))
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.perb5 = Perturbor(shape=(4,64,320,320))
        self.conv5a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.perb5a = Perturbor(shape=(4,64,320,320)) 
        self.conv6 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2) 

    
    def forward(self, x):
        # x = self.conv1_1(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.conv1_2(x)
        # x = F.leaky_relu(x, negative_slope=0.4)

        # x = self.convA2_1(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.convA2_2(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        # x = self.convA2_3(x)
        # x = F.leaky_relu(x, negative_slope=0.4)

        # x = self.conv3(x)
        # x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv4(x)
        x = self.perb4(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv4a(x)
        x = self.perb4a(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv5(x)
        x = self.perb5(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv5a(x)
        x = self.perb5a(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = F.upsample(x, scale_factor=2, mode='nearest')
        x = self.conv6(x)
        
        return x

class Perturbor(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(shape))
        self.W.requires_grad = True
        
        
    def forward(self, x):
        
        y = torch.tensor(1., dtype=float).cuda() + self.W
        x = torch.mul(x,y)

        return x