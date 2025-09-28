import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

import utils.detection_utils as utils
from model.minknet.detection_model import Model


class SparseGenerativeFeatureUpsampleNetwork(Model):

    def __init__(self, in_channels, upsample_feat_size=128, D=3, **kwargs):
        # upsample_feat_size: Feature size of dense upsample layer
        assert upsample_feat_size > 0
        # assert len(in_channels) == len(in_pixel_dists)
        super().__init__(D, **kwargs)
        self.OUT_PIXEL_DIST = 1
        self.network_initialization(in_channels, upsample_feat_size, D)
        self.weight_initialization()

    def network_initialization(self, in_channels, upsample_feat_size, D):
        up_kernel_size = 3
        self.conv_up1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels[0], in_channels[0], kernel_size=up_kernel_size, stride=2,
                 dimension=3),
            ME.MinkowskiBatchNorm(in_channels[0]),
            ME.MinkowskiELU())

        self.conv_up2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels[1], in_channels[0], kernel_size=up_kernel_size, stride=2,
                 dimension=3),
            ME.MinkowskiBatchNorm(in_channels[0]),
            ME.MinkowskiELU())

        self.conv_up3 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels[2], in_channels[1], kernel_size=up_kernel_size, stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(in_channels[1]),
            ME.MinkowskiELU())

        self.conv_up4 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels[3], in_channels[2], kernel_size=up_kernel_size, stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(in_channels[2]),
            ME.MinkowskiELU())

        self.conv_feat1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels[0], upsample_feat_size, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(upsample_feat_size),
            ME.MinkowskiELU())

        self.conv_feat2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels[1], upsample_feat_size, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(upsample_feat_size),
            ME.MinkowskiELU())

        self.conv_feat3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels[2], upsample_feat_size, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(upsample_feat_size),
            ME.MinkowskiELU())

        self.conv_feat4 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels[3], upsample_feat_size, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(upsample_feat_size),
            ME.MinkowskiELU())

        self.conv_cls1 = ME.MinkowskiConvolution(
            upsample_feat_size, 2, kernel_size=1, dimension=3)
        self.conv_cls2 = ME.MinkowskiConvolution(
            upsample_feat_size, 2, kernel_size=1, dimension=3)
        self.conv_cls3 = ME.MinkowskiConvolution(
            upsample_feat_size, 2, kernel_size=1, dimension=3)
        self.conv_cls4 = ME.MinkowskiConvolution(
            upsample_feat_size, 2, kernel_size=1, dimension=3)

        self.elu = ME.MinkowskiELU()
        self.pruning = ME.MinkowskiPruning()

    def forward(self, backbone_outputs, match_coords, is_train):
        # Enumerate network over pyramids.
        fpn_outputs = []
        targets = []
        classifications = []
        pyramid_output = None
        num_layers = len(backbone_outputs)
        if is_train:
            target_coords = [ME.utils.batched_coordinates([match[i][0] for match in match_coords])
                             for i in range(num_layers)]
            ambiguous_coords = [ME.utils.batched_coordinates([match[i][1] for match in match_coords])
                                for i in range(num_layers)]

        for layer_idx in reversed(range(num_layers)):
            conv_feat_layer = self.get_layer('conv_feat', layer_idx)
            conv_cls_layer = self.get_layer('conv_cls', layer_idx)
            conv_up_layer = self.get_layer('conv_up', layer_idx)

            # Current feature
            curr_feat = backbone_outputs[layer_idx]

            # Add previous layer output
            if pyramid_output is not None:
                assert pyramid_output.tensor_stride == curr_feat.tensor_stride
                curr_feat = curr_feat + pyramid_output

            # Two branches: upsample and fpn feature and classification
            # 1. FPN feature & classification
            fpn_feat = conv_feat_layer(curr_feat)
            feat_cls = conv_cls_layer(fpn_feat)
            pred_prob = F.softmax(feat_cls.F, 1)[:, 1]

            # target calculation
            target = None
            if is_train:
                target = torch.zeros(len(fpn_feat), dtype=torch.long)
                pos_ins = utils.map_coordinates(fpn_feat, torch.cat(ambiguous_coords[:layer_idx + 1]),
                                                force_stride=True)[0]
                target[pos_ins] = self.config.ignore_label
                pos_ins = utils.map_coordinates(fpn_feat, torch.cat(target_coords[:layer_idx + 1]),
                                                force_stride=True)[0]
                target[pos_ins] = 1

            # Get keep labels
            keep = (pred_prob > self.config.sfpn_min_confidence).cpu()
            if is_train:  # Force put GT labels within keep
                keep |= target == 1

            if torch.any(keep):
                # Prune and upsample
                pyramid_output = conv_up_layer(self.pruning(curr_feat, keep))
                # Generate final feature for current level
                final_pruned = self.pruning(fpn_feat, keep)
            else:
                pyramid_output = None
                final_pruned = None

            # Post processing
            classifications.insert(0, feat_cls)
            targets.insert(0, target)
            fpn_outputs.insert(0, final_pruned)

        print(fpn_outputs)
        print(len(fpn_outputs))
        print(targets)
        print(len(targets))
        print(fpn_outputs[0].size())
        print(targets[0].size())

        return fpn_outputs, targets, classifications
