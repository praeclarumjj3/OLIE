import torch.nn as nn
import torch.nn.functional as F
from modules.helpers.utils import normalize
from modules.olie_gan import OlieGAN

class OLIE(nn.Module):
    
    def __init__(self, solo, olie_gan):
        super().__init__()

        self.solo = solo
        self.olie_gan = olie_gan

    def forward(self, x):
        masks, images = self.solo(x)
        
        size = masks.shape[2]
        images = F.interpolate(images,(size,size))
        images = normalize(images)
        masks = F.tanh(masks)

        
        