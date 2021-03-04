import torch.nn as nn
import torch.nn.functional as F
import torch

class GatedConv2dWithActivation(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.4, inplace=True)):
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
    
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, batch_norm=True,activation=torch.nn.LeakyReLU(0.4, inplace=True)):
        super(GatedDeConv2dWithActivation, self).__init__()
        
        self.conv2d = GatedConv2dWithActivation(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        return self.conv2d(x)

class GatedEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 72, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(72)
        self.conv2 = nn.Conv2d(72, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)

        self.chan_conv = nn.Conv2d(144, 32, kernel_size=1, stride=1)

        self.cnum = 128

        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.gated_conv1 = GatedConv2dWithActivation(in_channels=32, out_channels=self.cnum, kernel_size=5, padding=2)
        self.gated_conv1A = GatedConv2dWithActivation(self.cnum, 2*self.cnum, 3, 1, padding=1)
        # downsample
        
        self.gated_conv2 = GatedConv2dWithActivation(2*self.cnum, 2*self.cnum, 3, 1, padding=1)
        self.gated_conv2A = GatedConv2dWithActivation(2*self.cnum, 4*self.cnum, 3, 1, padding=1)
        #downsample

        self.gated_conv3 = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        self.gated_conv3A = GatedConv2dWithActivation(4*self.cnum, 4*self.cnum, 3, 1, padding=1)

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

        feats = self.conv5(feats)

        return feats

class GatedDecoder(nn.Module):
    def __init__(self):
        super(GatedDecoder, self).__init__()

        self.cnum = 128

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

        feats = self.conv4E(feats)

        return feats

class RefineDecoder(nn.Module):
    def __init__(self):
        super(RefineDecoder, self).__init__()

        self.cnum = 32

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