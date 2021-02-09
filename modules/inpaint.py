import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

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

def visualize(x,layer):
    dim = int(x.shape[1])
    x = x[0].cpu() 
    x = x.permute(1, 2, 0).numpy()
    f, axarr = plt.subplots(int(dim**0.5),int(dim**0.5),figsize=(16,16))
    for j in range(x.shape[2]):
        r = int(j/dim**0.5)
        c = int(j%dim**0.5)
        axarr[r,c].imshow(x[:,:,j])
        axarr[r,c].axis('off')
    f.savefig('visualizations/{}.jpg'.format(layer))

class Inpainter(nn.Module):

    def __init__(self, hid_channels):
        super().__init__()

        self.encoder = Encoder(hid_channels)
        self.decoder = Decoder()

    def forward(self, recons, masks):
        x = self.encoder(recons,masks)
        x = self.decoder(x, recons, masks)

        return x

class Encoder(nn.Module):
    def __init__(self, hid_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, recons, mask):

        x = recons*(1-mask)
        x = torch.cat([x, mask], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.4)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        # dilated convolution blocks
        self.convA2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=8, dilation=8)
        self.convA2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=4, dilation=4)
        self.convA2_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1) 
        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) 
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)


    def forward(self, x, recons, masks):
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
        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv4a(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv5(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv5a(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv6(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv7(x)
        
        return x*masks + recons*(1-masks)