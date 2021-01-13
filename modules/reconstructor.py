import torch.nn as nn
import torch.nn.functional as F

class Reconstructor(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        # get the device of the model
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,padding=1, groups=12)
        self.p1 = nn.Conv2d(in_channels=in_channels,out_channels=int(in_channels/2),kernel_size=1, groups=12)
        self.bn1 = nn.BatchNorm2d(num_features=int(in_channels/2))
        self.conv2 = nn.Conv2d(in_channels=int(in_channels/2),out_channels=int(in_channels/2),kernel_size=3,padding=1, groups=12)
        self.p2 = nn.Conv2d(in_channels=int(in_channels/2),out_channels=int(in_channels/4),kernel_size=1, groups=12)
        self.bn2 = nn.BatchNorm2d(num_features=int(in_channels/4))
        self.conv3 = nn.Conv2d(in_channels=int(in_channels/4),out_channels=int(in_channels/4),kernel_size=3,padding=1, groups=12)
        self.p3 = nn.Conv2d(in_channels=int(in_channels/4),out_channels=3,kernel_size=1, groups=12)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, masks, images):
        x = self.upsample(masks)
        x = self.conv1(x)
        x = self.p1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.p2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.p3(x)
        x = F.relu(x)

        result = x
        return result