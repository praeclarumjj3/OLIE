import torch
from torch import nn
from torchvision.models import vgg16, vgg19

class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        vgg = vgg16(pretrained = True)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        features = list(vgg.features)[:23]
        # features: 3，8，15，22 :relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()
        self.layers = {'3': 'relu1_2', '8': 'relu2_2', '15': 'relu3_3', '22': 'relu4_3'}
        
    def forward(self, x):
        results = {}
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22}:
                results[self.layers[str(ii)]] = x
        
        return results

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        vgg = vgg19(pretrained = True)
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                vgg.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        features = list(vgg.features)[:27]
        # features: 3，8，17，26 :relu1_2,relu2_2,relu3_4,relu4_4
        self.features = nn.ModuleList(features).eval()
        self.layers = {'3': 'relu1_2', '8': 'relu2_2', '17': 'relu3_4', '26': 'relu4_4'}
        
    def forward(self, x):
        results = {}
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,17,26}:
                results[self.layers[str(ii)]] = x
        
        return results

class VGG_head(nn.Module):
    def __init__(self):
        super(VGG_head, self).__init__()
        self.vgg = Vgg19().cuda()
        
    def forward(self, gen_imgs, gt_imgs):
        output_feature = self.vgg(gen_imgs)
        target_feature = self.vgg(gt_imgs)
        return output_feature, target_feature

class VGGLoss(nn.Module):
    def __init__(self, n_gpus=1, masked=False):
        super().__init__()
        self.masked = masked
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.vgg_head = VGG_head()
        # self.vgg_list = [Vgg16(requires_grad=False) for i in range(n_gpus)]

    def vgg_loss(self, gen_imgs, gt_imgs):
        output_feature, target_feature = self.vgg_head(gen_imgs,gt_imgs)
        loss = (self.l1_loss(output_feature['relu1_2'], target_feature['relu1_2']) 
            + self.l1_loss(output_feature['relu2_2'], target_feature['relu2_2'])
#             + self.l1_loss(output_feature['relu3_4'], target_feature['relu3_4'])
#             + self.l1_loss(output_feature['relu4_4'], target_feature['relu4_4'])
        )
        return loss

    def forward(self, gen_imgs, gt_imgs, masks=None):
        if self.masked:
            gen_imgs = masks * gen_imgs
            gt_imgs = masks * gt_imgs
        # Note: It can be batch-lized
        # mean_image_loss = []
        # for frame_idx in range(targets.size(1)):
        #     mean_image_loss.append(
        #         self.vgg_loss(outputs[:, frame_idx], targets[:, frame_idx])
        #     )
        # mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        mean_image_loss = self.vgg_loss(gen_imgs, gt_imgs)
        return mean_image_loss

class ReconLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction=reduction)
        
    def forward(self, gen_imgs, gt_imgs):
        return self.loss_fn(gen_imgs, gt_imgs)