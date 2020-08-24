# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder2(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder2, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            #self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            setattr(self, "upconv_{}_0".format(i), ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            setattr(self, "upconv_{}_1".format(i), ConvBlock(num_ch_in, num_ch_out))

        for s in self.scales:
            #self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            setattr(self, "dispconv_{}".format(s), Conv3x3(self.num_ch_dec[s], self.num_output_channels))
            setattr(self, "dispconv_r{}".format(s), Conv3x3(32,1))
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

        # self.model = nn.Sequential(ConvBlock(4,16),
        #                     ConvBlock(16,32),
        #                     ConvBlock(32,32),
        #                     ConvBlock(32,64),
        #                     ConvBlock(64,64),
        #                     ConvBlock(64,128),
        #                     ConvBlock(128,128),
        #                     ConvBlock(128,64),
        #                     ConvBlock(64,64),
        #                     ConvBlock(64,32))

    def forward(self, input_features):
        self.outputs = {}
        # gt_mask = gt.sign()
        # gt_median = torch.median(gt[gt>0])
        # all_scale = gt / gt_median
        # gt[all_scale>2] = 0
        # gt[all_scale<0.5]=0

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = getattr(self, "upconv_{}_0".format(i))(x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = getattr(self, "upconv_{}_1".format(i))(x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(getattr(self, "dispconv_{}".format(i))(x))
                # blur_depth = self.sigmoid(getattr(self, "dispconv_{}".format(i))(x))
                # blur_depth = F.interpolate(blur_depth, [h, w], mode="bilinear", align_corners=False)
                
                # blur_depth = gt_mask * gt + (1-gt_mask) * blur_depth
                # print(gt_mask.size())
                # blur_rgbd = torch.cat((rgb,blur_depth),1)
                # ref_rgbd = self.model(blur_rgbd)
                # self.outputs[("disp", i)] = self.sigmoid(getattr(self, "dispconv_r{}".format(i))(ref_rgbd))

        return self.outputs
