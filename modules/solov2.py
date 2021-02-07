# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.modeling.backbone import build_backbone
from detectron2.structures import ImageList

from adet.config import get_cfg
from modules.solov2_heads import SOLOv2InsHead, SOLOv2MaskHead
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

__all__ = ["SOLOv2"]


class SOLOv2(nn.Module):
    """
    SOLOv2 model. Creates FPN backbone, instance branch for kernels and categories prediction,
    mask branch for unified mask features.
    Calculates and applies proper losses to class and masks.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.scale_ranges = cfg.MODEL.SOLOV2.FPN_SCALE_RANGES
        self.strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.sigma = cfg.MODEL.SOLOV2.SIGMA
        
        # Instance parameters.
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS

        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS

        # Mask parameters.
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS

        # build the backbone.
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()

        # build the ins head.
        instance_shapes = [backbone_shape[f] for f in self.instance_in_features]
        self.ins_head = SOLOv2InsHead(cfg, instance_shapes)

        # build the mask head.
        mask_shapes = [backbone_shape[f] for f in self.mask_in_features]
        self.mask_head = SOLOv2MaskHead(cfg, mask_shapes)

        # image transform
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
    
        # ins branch
        ins_features = [features[f] for f in self.instance_in_features]
        ins_features = self.split_feats(ins_features)
        kernel_pred = self.ins_head(ins_features)

        # mask branch
        mask_features = [features[f] for f in self.mask_in_features]
        
        mask_pred = self.mask_head(mask_features)
        results = self.inference(kernel_pred, mask_pred, batched_inputs)
#         x = results[0].cpu() 
#         x = x.permute(1, 2, 0).numpy()
#         f, axarr = plt.subplots(12,12,figsize=(16,16))
#         for j in range(x.shape[2]):
#             r = int(j/12)
#             c = j%12
#             axarr[r,c].imshow(x[:,:,j])
#             axarr[r,c].axis('off')
#         f.savefig('visualizations/x_m.jpg')
#         print(results[0])
#         exit()
        return results, torch.stack(batched_inputs,dim=0)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x.to(self.device) for x in batched_inputs]
        norms = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(norms, self.backbone.size_divisibility)
        return images


    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    def inference(self, pred_kernels, pred_masks, images):

        results = []
        for img_idx in range(len(images)):

            # prediction.
            pred_kernel = [pred_kernels[len(pred_kernels)-1][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_kernel = torch.cat(pred_kernel, dim=0)

            # inference for single image.
            result = self.inference_single_image(pred_kernel, pred_mask)
            results.append(result)
        return torch.stack(results,0)

    def inference_single_image(
            self, kernel_preds, seg_preds
    ):
        # mask encoding.
        N, I = kernel_preds.shape
        kernel_preds = kernel_preds.view(N, I, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()
        return seg_preds
