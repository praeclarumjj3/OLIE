# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
import math
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.visualizer import ColorMode, Visualizer
from .utils import point_nms, mask_nms, matrix_nms
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
        
        # Inference parameters.
        self.max_before_nms = cfg.MODEL.SOLOV2.NMS_PRE
        self.score_threshold = cfg.MODEL.SOLOV2.SCORE_THR
        self.update_threshold = cfg.MODEL.SOLOV2.UPDATE_THR
        self.mask_threshold = cfg.MODEL.SOLOV2.MASK_THR
        self.max_per_img = cfg.MODEL.SOLOV2.MAX_PER_IMG
        self.nms_kernel = cfg.MODEL.SOLOV2.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.SOLOV2.NMS_SIGMA
        self.nms_type = cfg.MODEL.SOLOV2.NMS_TYPE

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
        images, size = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
    
        # ins branch
        ins_features = [features[f] for f in self.instance_in_features]
        ins_features = self.split_feats(ins_features)
        cate_pred, kernel_pred = self.ins_head(ins_features)

        # mask branch
        mask_features = [features[f] for f in self.mask_in_features]
        mask_pred = self.mask_head(mask_features)

        cate_pred = [point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                         for cate_p in cate_pred]
        # do inference for results.
        results = self.inference_instance_maps(cate_pred, kernel_pred, mask_pred, images.image_sizes, size, batched_inputs)

        # plt.rcParams.update({'font.size': 3})
        # results = torch.stack(results,0)
        # x = results[0]
        # dim = int(x.shape[0])
        # x = x.cpu() 
        # x = x.permute(1, 2, 0).numpy()
        # f, axarr = plt.subplots(int(dim**0.5)+1,int(dim**0.5)+1,figsize=(16,16))
        # for j in range(dim):
        #     r = int(j/int((dim**0.5)+1))
        #     c = int(j%int((dim**0.5)+1))
        #     axarr[r,c].imshow(x[:,:,j])
        #     axarr[r,c].axis('off')
        # f.savefig('visualizations/{}.jpg'.format('solo'))
        # exit()
        
        return results

        # if len(batched_inputs) == 1:
        #     return results, batched_inputs[0].unsqueeze(0) 
        
        # return results, torch.stack(batched_inputs,dim=0)


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x.to(self.device) for x in batched_inputs]
        norms = [self.normalizer(x) for x in images]
        size = (norms[0].shape[1],norms[0].shape[2])
        images = ImageList.from_tensors(norms, self.backbone.size_divisibility)
        return images, size

    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))

    
    def inference_instance_maps(self, pred_cates, pred_kernels, pred_masks, cur_sizes, ori_size, images):
        assert len(pred_cates) == len(pred_kernels)

        results = []
        num_ins_levels = len(pred_cates)
        for img_idx in range(len(images)):

            # prediction.
            pred_cate = [pred_cates[i][img_idx].view(-1, self.num_classes).detach()
                          for i in range(num_ins_levels)]
            pred_kernel = [pred_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()
                            for i in range(num_ins_levels)]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)

            # inference for single image.
            result = self.get_instance_maps(pred_cate, pred_kernel, pred_mask,
                                                 cur_sizes[img_idx], ori_size)
            results.append(result)
        
        return results

        # if len(results) == 1:
        #     return results[0].unsqueeze(0) 
        
        # return torch.stack(results,dim=0)


    def get_instance_maps(
            self, cate_preds, kernel_preds, seg_preds, cur_size, ori_size
    ):
        # overall info.
        h, w = cur_size
        f_h, f_w = seg_preds.size()[-2:]
        ratio = math.ceil(h/f_h)
        upsampled_size_out = (int(f_h*ratio), int(f_w*ratio))
        
        # process.
        inds = (cate_preds > self.score_threshold)
        cate_scores = cate_preds[inds]
        
        if len(cate_scores) == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        # cate_labels & kernel_preds
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        kernel_preds = kernel_preds[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = kernel_preds.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.instance_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.instance_strides[ind_]
        strides = strides[inds[:, 0]]

        # mask encoding.
        N, I = kernel_preds.shape
        kernel_preds = kernel_preds.view(N, I, 1, 1)
        seg_preds = F.conv2d(seg_preds, kernel_preds, stride=1).squeeze(0).sigmoid()

        # mask.
        seg_masks = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # mask scoring.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[:self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        if self.nms_type == "matrix":
            # matrix nms & filter.
            cate_scores = matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                          sigma=self.nms_sigma, kernel=self.nms_kernel)
            keep = cate_scores >= self.update_threshold
        elif self.nms_type == "mask":
            # original mask nms.
            keep = mask_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                 nms_thr=self.mask_threshold)
        else:
            raise NotImplementedError

        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # reshape to original size.
        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_size,
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > self.mask_threshold

        results = Instances(ori_size)
        results.pred_classes = cate_labels
        results.scores = cate_scores
        results.pred_masks = seg_masks

        # get bbox from mask
        pred_boxes = torch.zeros(seg_masks.size(0), 4)
        for i in range(seg_masks.size(0)):
           mask = seg_masks[i].squeeze()
           ys, xs = torch.where(mask)
           pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
        results.pred_boxes = Boxes(pred_boxes)

#         print(results.pred_boxes)
#         print(results.pred_classes)
#         print(results.scores)
#         exit()

        return results.pred_masks
