import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

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
    for j in range(x.shape[2]):
        r = int(j/dim**0.5)
        c = int(j%dim**0.5)
        axarr[r,c].imshow(x[:,:,j])
        axarr[r,c].axis('off')
    f.savefig('visualizations/{}.jpg'.format(layer))

def visualize_image(image,x,layer):
    plt.rcParams.update({'font.size': 3})
    dim = int(x.shape[1])
    x = x[0].cpu() 
    f, axarr = plt.subplots(int(dim**0.5),int(dim**0.5),figsize=(16,16))
    for j in range(x.shape[0]):
        r = int(j/dim**0.5)
        c = int(j%dim**0.5)
        i = image.squeeze(0).cpu() * x[j]
        i = torch.clamp(torch.round(i),min=0., max = 255.) * torch.tensor(1./255)
        i = torch.where(i>0.5,1.,0.)
        i = image.squeeze(0).cpu()*i
        i = torch.clamp(torch.round(i),min=0., max = 255.) * torch.tensor(1./255)
        i = i.permute(1, 2, 0).numpy()
        axarr[r,c].imshow(i[:,:,:])
        axarr[r,c].axis('off')
    f.savefig('visualizations/{}.jpg'.format(layer))

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
        self.conv2 = nn.Conv2d(72, 144, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(144)

        self.mask_conv1 = nn.Conv2d(in_channels,256,kernel_size=1,stride=1)
        self.mask_conv2 = nn.Conv2d(256,512,kernel_size=1,stride=1)
        
    def forward(self, x, maps):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.4)

        ins = maps*x

        masks = self.mask_conv1(ins)
        masks = F.leaky_relu(masks, negative_slope=0.4)
        masks = self.mask_conv2(masks)
        masks = F.leaky_relu(masks, negative_slope=0.4)

        return masks

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.mask_conv1 = nn.Conv2d(512,256,kernel_size=1,stride=1)
        self.mask_conv2 = nn.Conv2d(256,64,kernel_size=1,stride=1)
        self.mask_conv3 = nn.Conv2d(64,32,kernel_size=1,stride=1)
        self.mask_conv4 = nn.Conv2d(32,3,kernel_size=1,stride=1)


    def forward(self, masks):
       
        masks = self.mask_conv1(masks)
        masks = F.leaky_relu(masks, negative_slope=0.4)
        masks = self.mask_conv2(masks)
        masks = F.leaky_relu(masks, negative_slope=0.4)
        masks = F.upsample(masks, scale_factor=4, mode='bilinear', align_corners=False)
        masks = self.mask_conv3(masks)
        masks = F.leaky_relu(masks, negative_slope=0.4)
        masks = self.mask_conv4(masks)
        
        return masks