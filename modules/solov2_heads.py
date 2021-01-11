# -*- coding: utf-8 -*-
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec

class SOLOv2InsHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Instance Head.
        """
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS
        # Convolutions to use in the towers
        self.type_dcn = cfg.MODEL.SOLOV2.TYPE_DCN
        self.num_levels = len(self.instance_in_features)
        assert self.num_levels == len(self.instance_strides), \
            print("Strides should match the features.")
        # fmt: on

        head_configs = {"kernel": (cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS,
                                   cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE,
                                   cfg.MODEL.SOLOV2.USE_COORD_CONV)
                        }

        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS, \
            print("In channels should equal to tower in channels!")

        for head in head_configs:
            tower = []
            num_convs, use_deformable, use_coord = head_configs[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                if i == 0:
                    if use_coord:
                        chn = self.instance_in_channels + 2
                    else:
                        chn = self.instance_in_channels
                else:
                    chn = self.instance_channels

                tower.append(conv_func(
                        chn, self.instance_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, self.instance_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.kernel_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [
             self.kernel_tower,
             self.kernel_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        kernel_pred = []

        for idx, feature in enumerate(features):
            ins_kernel_feat = feature
            # concat coord
            x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
            y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')

            # kernel
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))

    
        return kernel_pred


class SOLOv2MaskHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Mask Head.
        """
        super().__init__()
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.mask_in_features), \
            print("Input shape should match the features.")
        # fmt: on
        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, self.mask_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            nn.GroupNorm(32, self.num_masks),
            nn.ReLU(inplace=True)
        )

        for modules in [self.convs_all_levels, self.conv_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.num_levels):
            mask_feat = features[i]
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
                y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            feature_add_all_level += self.convs_all_levels[i](mask_feat)

        mask_pred = self.conv_pred(feature_add_all_level)
        return mask_pred