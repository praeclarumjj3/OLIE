import torch.nn as nn
import torch.nn.functional as F
from gated_modules import Refiner, GatedEncoder, GatedDecoder 
from utils import masking, visualize, masking_threshold, masking_objects, normalize, mask_shuffle

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

#         self.chan_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        
        self.cnum = 256

        self.enc_conv1 = nn.Conv2d(in_channels,self.cnum,kernel_size=3,stride=1,padding=1)
        self.enc_conv1A = nn.Conv2d(self.cnum, 2*self.cnum, 3, 1, padding=1)

        # self.enc_conv2 = nn.Conv2d(2*self.cnum, 2*self.cnum, 3, 1, padding=1)
        # self.enc_conv2A = nn.Conv2d(2*self.cnum, 4*self.cnum, 3, 1, padding=1)

        # self.enc_conv3 = nn.Conv2d(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        # self.enc_conv3A = nn.Conv2d(4*self.cnum, 4*self.cnum, 3, 1, padding=1)

        # self.conv5 = nn.Conv2d(4*self.cnum, 4*self.cnum, 3, 1, padding=1)
        
    def forward(self, x, maps):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.4)

        ins = maps*x

#         ins = self.chan_conv(ins)
        
#         visualize(ins,'ins')

        feats = self.enc_conv1(ins)
        feats = F.leaky_relu(feats, negative_slope=0.4)
#         visualize(feats,'feats1')
        feats = self.enc_conv1A(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
#         visualize(feats,'feats2')
        
        # feats = self.enc_conv2(feats)
        # feats = F.leaky_relu(feats, negative_slope=0.4)
        # feats = self.enc_conv2A(feats)
        # feats = F.leaky_relu(feats, negative_slope=0.4)
        
        # feats = self.enc_conv3(feats)
        # feats = F.leaky_relu(feats, negative_slope=0.4)
        # feats = self.enc_conv3A(feats)
        # feats = F.leaky_relu(feats, negative_slope=0.4)

        # feats = self.conv5(feats)
        # feats = F.leaky_relu(feats, negative_slope=0.4)

        return feats

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.cnum = 256

        # upsample
        # self.conv1 = nn.Conv2d(4*self.cnum, 2*self.cnum, 3, 1, padding=1)
        self.conv1A = nn.Conv2d(2*self.cnum, self.cnum, 3, 1, padding=1)
        
        #upsample
        self.conv2 = nn.Conv2d(self.cnum, self.cnum//2, 3, 1, padding=1)
        self.conv2A = nn.Conv2d(self.cnum//2, self.cnum//4, 3, 1, padding=1)
        
        self.conv3 = nn.Conv2d(self.cnum//4, 3, 3, 1, padding=1)


    def forward(self, feats):
       
        # feats = self.conv1(feats)
        # feats = F.leaky_relu(feats, negative_slope=0.4)
        feats = F.interpolate(feats, scale_factor=2)
        feats = self.conv1A(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
#         visualize(feats,'dec_feats1')

        feats = self.conv2(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
        feats = F.interpolate(feats, scale_factor=2)
        feats = self.conv2A(feats)
        feats = F.leaky_relu(feats, negative_slope=0.4)
#         visualize(feats,'dec_feats2')

        feats = self.conv3(feats)
        
        return feats