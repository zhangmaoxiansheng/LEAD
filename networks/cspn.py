import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from layers import *
import torchvision.models as models
from collections import OrderedDict
from .resnet_encoder import ResnetEncoder,resnet_multiimage_input
class CSPN_Propagate(nn.Module):
    def __init__(self,crop_h,crop_w):
        super(CSPN_Propagate, self).__init__()
        self.crop_h = crop_h#[96,128,160,192]
        self.crop_w = crop_w#[192,256,384,640]
        self.model_ref0 = ConvBlock(16,8)
        #self.get_disp0 = nn.Sequential(Conv3x3(32,1),nn.Sigmoid())
        #self.ref_disp0 = Conv3x3(32,8)
        #self.models0 = nn.ModuleList([self.model_ref0,self.get_disp0,self.ref_disp0])

        self.model_ref1 = ConvBlock(16,8)
        # self.get_disp1 = nn.Sequential(Conv3x3(32,1),nn.Sigmoid())
        # self.ref_disp1 = Conv3x3(32,8)
        # self.models1 = nn.ModuleList([self.model_ref1,self.get_disp1,self.ref_disp1])
        
        self.model_ref2 = ConvBlock(16,8)
        # self.get_disp2 = nn.Sequential(Conv3x3(32,1),nn.Sigmoid())
        # self.ref_disp2 = Conv3x3(32,8)
        # self.models2 = nn.ModuleList([self.model_ref2,self.get_disp2,self.ref_disp2])
        
        self.model_ref3 = ConvBlock(16,8)
                            # ConvBlock(32,64),
                            # ConvBlock(64,32),
                            # ConvBlock(32,32))
        # self.get_disp3 = nn.Sequential(Conv3x3(32,1),nn.Sigmoid())
        # self.ref_disp3 = Conv3x3(32,8)
        # self.models3 = nn.ModuleList([self.model_ref3,self.get_disp3,self.ref_disp3])

        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3])
        #self.sigmoid = nn.Sigmoid()
        self.cspn = Affinity_Propagate()
    def center_crop(self,image,h=160,w=320):
        #mask = torch.zeros_like(image)
        origin_h = image.size(2)
        origin_w = image.size(3)
        
        h_start = max(int(round((origin_h-h)/2)),0)
        h_end = min(h_start + h,origin_h)
        w_start = max(int(round((origin_w-w)/2)),0)
        w_end = min(w_start + w,origin_w)
        output = image[:,:,h_start:h_end,w_start:w_end] 
        #mask[:,:,h_start:h_end,w_start:w_end] = 1
        return output
    #small feature 256 -> fuse small gt -> conv -> larger area -> next stage
    # TODO finish the stage forward 
    def stage_forward(self,features,rgbd,dep_last,model,h,w):
        #dep_last is the padded depth
        rgbd = self.center_crop(rgbd,h,w)
        feature_crop = self.center_crop(features,h,w)
        dep = rgbd[:,0,:,:].unsqueeze(1)
        
        mask = dep_last.sign()
        dep_fusion = dep_last * mask + dep * (1-mask) 
        feature_stage = feature_crop
        #feature_stage = torch.cat((feature_crop,dep_fusion),1)

        # model_feature = model[0]
        # model_get_disp = model[1]
        # model_guided_map = model[2]
        
        # x = model_feature(feature_stage)
        # #disp_init = model_get_disp(x)
        # disp_init = dep_fusion
        # guided_map = model_guided_map(x)
        # x = self.cspn(guided_map,disp_init,dep_last)

        guided_map = model(feature_stage)
        x = self.cspn(guided_map, dep_fusion, dep_last)
        x = torch.clamp(x,0,1)
        return x
    
    def stage_pad(self,depth,h,w):
        hs = depth.size(2)
        ws = depth.size(3)
        pad_w = (w-ws) // 2
        pad_h= (h-hs) // 2
        pad = nn.ZeroPad2d((pad_w,pad_w,pad_h,pad_h))
        depth_pad = pad(depth)
        return depth_pad

    def stage_block(self,features,rgbd,dep_last,stage,outputs):
        for index, i in enumerate(stage):
            if index == len(stage) - 1:#the last stage
                outputs[("disp",i)] = self.stage_forward(features,rgbd,dep_last,self.models[i],self.crop_h[i],self.crop_w[i])
            else:
                with torch.no_grad():
                    ref_last = self.stage_forward(features,rgbd,dep_last,self.models[i],self.crop_h[i],self.crop_w[i])
                    dep_last = self.stage_pad(ref_last,self.crop_h[i+1],self.crop_w[i+1])
                    outputs[("disp",i)] = ref_last
        return outputs

    def forward(self,features, blur_depth, gt, rgb, stage):

        outputs = {}
        scale = torch.median(gt[gt>0]) / torch.median(blur_depth[gt>0])
        all_scale = gt / blur_depth
        blur_depth_o = blur_depth * scale
        
        gt[all_scale>1.2] = 0
        gt[all_scale<0.8]=0
        gt_mask = gt.sign()
        blur_depth = gt_mask * gt + (1-gt_mask) * blur_depth_o
        rgbd = torch.cat((rgb, blur_depth_o),1)
        dep_0 = self.center_crop(gt,96,192)

        self.stage_block(features,rgbd,dep_0, stage, outputs)
        outputs["blur_disp"] = blur_depth_o
        outputs["disp_all_in"] = blur_depth
        outputs['scale'] = scale
        return outputs


class Affinity_Propagate(nn.Module):

    def __init__(self,
                 prop_time=3,
                 prop_kernel=3,
                 norm_type='8sum',
                 post_process = False):
        """

        Inputs:
            prop_time: how many steps for CSPN to perform
            prop_kernel: the size of kernel (current only support 3x3)
            way to normalize affinity
                '8sum': normalize using 8 surrounding neighborhood
                '8sum_abs': normalization enforcing affinity to be positive
                            This will lead the center affinity to be 0
        """
        super(Affinity_Propagate, self).__init__()
        self.post_process = post_process
        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'

        self.norm_type = norm_type
        assert norm_type in ['8sum', '8sum_abs']

        self.in_feature = 1
        self.out_feature = 1
        self.sigmoid = nn.Sigmoid()

        if self.post_process:
            self.model_ref = nn.Sequential(ConvBlock(1,8),
                            ConvBlock(8,4),
                            ConvBlock(4,1),
                            nn.Sigmoid())


    def forward(self, guidance, blur_depth, sparse_depth=None):

        self.sum_conv = nn.Conv3d(in_channels=8,
                                  out_channels=1,
                                  kernel_size=(1, 1, 1),
                                  stride=1,
                                  padding=0,
                                  bias=False)
        weight = torch.ones(1, 8, 1, 1, 1).cuda()
        self.sum_conv.weight = nn.Parameter(weight)
        for param in self.sum_conv.parameters():
            param.requires_grad = False

        gate_wb, gate_sum = self.affinity_normalization(guidance)

        # pad input and convert to 8 channel 3D features
        raw_depth_input = blur_depth

        #blur_depht_pad = nn.ZeroPad2d((1,1,1,1))
        result_depth = blur_depth

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()#not correct! outputs is 1 0 -1
            #result_depth = (1 - sparse_mask) * result_depth + sparse_mask * sparse_depth

        for i in range(self.prop_time):
            # one propagation
            if sparse_depth is not None and i == 0:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * sparse_depth

            spn_kernel = self.prop_kernel
            result_depth = self.pad_blur_depth(result_depth)
            neigbor_weighted_sum = self.sum_conv(gate_wb * result_depth)
            neigbor_weighted_sum = neigbor_weighted_sum.squeeze(1)
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum

            if '8sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            else:
                raise ValueError('unknown norm %s' % self.norm_type)

            # if sparse_depth is not None:
            #     result_depth = (1 - sparse_mask) * result_depth + sparse_mask * sparse_depth
        if self.post_process:
            result_depth = self.model_ref(result_depth)

        return result_depth

    def affinity_normalization(self, guidance):

        # normalize features
        if 'abs' in self.norm_type:
            guidance = torch.abs(guidance)

        gate1_wb_cmb = guidance.narrow(1, 0                   , self.out_feature)
        gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
        gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
        gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
        gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
        gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
        gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
        gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        gate1_wb_cmb = left_top_pad(gate1_wb_cmb).unsqueeze(1)

        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        gate2_wb_cmb = center_top_pad(gate2_wb_cmb).unsqueeze(1)

        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        gate3_wb_cmb = right_top_pad(gate3_wb_cmb).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        gate4_wb_cmb = left_center_pad(gate4_wb_cmb).unsqueeze(1)

        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        gate5_wb_cmb = right_center_pad(gate5_wb_cmb).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        gate6_wb_cmb = left_bottom_pad(gate6_wb_cmb).unsqueeze(1)

        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        gate7_wb_cmb = center_bottom_pad(gate7_wb_cmb).unsqueeze(1)

        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        gate8_wb_cmb = right_bottm_pad(gate8_wb_cmb).unsqueeze(1)

        gate_wb = torch.cat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
                             gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = torch.abs(gate_wb)
        abs_weight = self.sum_conv(gate_wb_abs)

        gate_wb = torch.div(gate_wb, abs_weight)
        gate_sum = self.sum_conv(gate_wb)

        gate_sum = gate_sum.squeeze(1)
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, gate_sum


    def pad_blur_depth(self, blur_depth):
        # top pad
        left_top_pad = nn.ZeroPad2d((0,2,0,2))
        blur_depth_1 = left_top_pad(blur_depth).unsqueeze(1)
        center_top_pad = nn.ZeroPad2d((1,1,0,2))
        blur_depth_2 = center_top_pad(blur_depth).unsqueeze(1)
        right_top_pad = nn.ZeroPad2d((2,0,0,2))
        blur_depth_3 = right_top_pad(blur_depth).unsqueeze(1)

        # center pad
        left_center_pad = nn.ZeroPad2d((0,2,1,1))
        blur_depth_4 = left_center_pad(blur_depth).unsqueeze(1)
        right_center_pad = nn.ZeroPad2d((2,0,1,1))
        blur_depth_5 = right_center_pad(blur_depth).unsqueeze(1)

        # bottom pad
        left_bottom_pad = nn.ZeroPad2d((0,2,2,0))
        blur_depth_6 = left_bottom_pad(blur_depth).unsqueeze(1)
        center_bottom_pad = nn.ZeroPad2d((1,1,2,0))
        blur_depth_7 = center_bottom_pad(blur_depth).unsqueeze(1)
        right_bottm_pad = nn.ZeroPad2d((2,0,2,0))
        blur_depth_8 = right_bottm_pad(blur_depth).unsqueeze(1)

        result_depth = torch.cat((blur_depth_1, blur_depth_2, blur_depth_3, blur_depth_4,
                                  blur_depth_5, blur_depth_6, blur_depth_7, blur_depth_8), 1)
        return result_depth


    def normalize_gate(self, guidance):
        gate1_x1_g1 = guidance.narrow(1,0,1)
        gate1_x1_g2 = guidance.narrow(1,1,1)
        gate1_x1_g1_abs = torch.abs(gate1_x1_g1)
        gate1_x1_g2_abs = torch.abs(gate1_x1_g2)
        elesum_gate1_x1 = torch.add(gate1_x1_g1_abs, gate1_x1_g2_abs)
        gate1_x1_g1_cmb = torch.div(gate1_x1_g1, elesum_gate1_x1)
        gate1_x1_g2_cmb = torch.div(gate1_x1_g2, elesum_gate1_x1)
        return gate1_x1_g1_cmb, gate1_x1_g2_cmb


    def max_of_4_tensor(self, element1, element2, element3, element4):
        max_element1_2 = torch.max(element1, element2)
        max_element3_4 = torch.max(element3, element4)
        return torch.max(max_element1_2, max_element3_4)

    def max_of_8_tensor(self, element1, element2, element3, element4, element5, element6, element7, element8):
        max_element1_2 = self.max_of_4_tensor(element1, element2, element3, element4)
        max_element3_4 = self.max_of_4_tensor(element5, element6, element7, element8)
        return torch.max(max_element1_2, max_element3_4)