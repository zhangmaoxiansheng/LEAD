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
import numpy as np
import pdb
#from .deform_conv import DeformConv
class Iterative_Propagate(nn.Module):
    def __init__(self,crop_h,crop_w,mode='c',dropout=False):
        super(Iterative_Propagate, self).__init__()
        self.crop_mode = mode
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.model_ref0 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref1 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref2 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref3 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,2),
                            ConvBlock(32,64,4),
                            ConvBlock(32,32,2),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref4 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,2),
                            ConvBlock(32,64,4),
                            ConvBlock(32,32,2),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())

        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3,self.model_ref4])
        #self.dep_enc = Depth_encoder()
        self.propagate_time = 1
        self.dropout = dropout
    
    def crop(self,image,h=160,w=320):
        origin_h = image.size(2)
        origin_w = image.size(3)
        if self.crop_mode=='b':
            origin_h = image.size(2)
            origin_w = image.size(3)
            h_start = max(int(round(origin_h-h)),0)
            w_start = max(int(round((origin_w-w)/2)),0)
            w_end = min(w_start + w,origin_w)
            output = image[:,:,h_start:,w_start:w_end]
        else:
            h_start = max(int(round((origin_h-h)/2)),0)
            h_end = min(h_start + h,origin_h)
            w_start = max(int(round((origin_w-w)/2)),0)
            w_end = min(w_start + w,origin_w)
            output = image[:,:,h_start:h_end,w_start:w_end] 
        return output
    
    def stage_pad(self,depth,h,w):
        if self.crop_mode == 'b':
            hs = depth.size(2)
            ws = depth.size(3)
            pad_w = (w-ws) // 2
            pad_h= h-hs
            pad = nn.ZeroPad2d((pad_w,pad_w,pad_h,0))
            depth_pad = pad(depth)
        else:
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
                outputs[("disp",i)], outputs[("condition",i)] = self.stage_forward(features,rgbd,dep_last,i)
            else:
                #with torch.no_grad():
                outputs[("disp",i)], outputs[("condition",i)] = self.stage_forward(features,rgbd,dep_last,i)
                dep_last = self.stage_pad(outputs[("disp",i)],self.crop_h[i+1],self.crop_w[i+1])
                #outputs[("dep_last",i)] = dep_last

        return outputs
    def scale_adjust(self,gt,dep):
        if torch.median(dep[gt>0])>0:
            scale = torch.median(gt[gt>0]) / torch.median(dep[gt>0])
        else:
            scale = 1
        dep_ref = dep * scale
        return dep_ref,scale
    
    
    def stage_forward(self,features,rgbd,dep_last,stage):
        if stage > 2:
            model = self.models[0]
        else:
            model = self.models[1]
        #dep_enc = self.dep_enc
        h = self.crop_h[stage]
        w = self.crop_w[stage]
        #dep_last is the padded depth
        rgbd = self.crop(rgbd,h,w)
        feature_crop = self.crop(features,h,w)
        dep_gt = self.crop(self.gt,h,w)
        dep = rgbd[:,3,:,:].unsqueeze(1)
        if dep[dep_last>0].shape != torch.Size([0]):
            if torch.median(dep[dep_last>0]) > 0:
                scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
            else:
                scale = 1
        else:
            scale = 1
            print("warning dep[dep_last>0] is empty,stage is %d"%stage)
        dep = dep * scale
        mask = dep_last.sign()
        mask_gt = dep_gt.sign()
        dep_fusion = dep_last * mask + dep * (1-mask)
        dep_fusion = dep_gt * mask_gt + dep_fusion * (1-mask_gt)
        feature_stage = torch.cat((feature_crop,dep_fusion),1)
        dep = model(feature_stage)
            # if torch.median(dep[dep_last>0]) > 0:
            #     scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
            # else:
            #     scale = 1
            # dep = dep * scale
        return dep, feature_stage
    
    def eval_step(self, features, blur_depth, gt_part, rgb, stage, dep_last,depth_gt):
        gt = gt_part.clone()
        all_scale = gt / blur_depth
        blur_depth_o, scale = self.scale_adjust(gt,blur_depth)
        gt_crop = self.crop(gt,self.crop_h[stage],self.crop_w[stage])
        gt[all_scale>1.2] = 0
        gt[all_scale<0.8]=0
        #gt_crop = self.crop(gt,self.crop_h[stage],self.crop_w[stage])
        if stage == 0:
            dep_last = self.crop(gt,self.crop_h[stage],self.crop_w[stage])
        
        rgbd = torch.cat((rgb, blur_depth_o),1)
        
        if stage != 0:
            dep_last_ = self.stage_pad(dep_last,self.crop_h[stage],self.crop_w[stage])
        else:
            dep_last_ = dep_last
        
        stage_out, _ = self.stage_forward(features,rgbd,dep_last_,stage)
        #to depth
        _,dep_last_depth = disp_to_depth(dep_last,0.9,100)
        _,stage_out_depth = disp_to_depth(stage_out,0.9,100)
        gt_crop_depth = self.crop(depth_gt,self.crop_h[stage],self.crop_w[stage])
        mask2 = gt_crop_depth > 0
        mask = dep_last_depth > 0
        if stage == 0:
            error = (stage_out_depth[mask2]-gt_crop_depth[mask2]).abs().mean()
        else:
            stage_out_depth_crop = self.crop(stage_out_depth,self.crop_h[stage-1],self.crop_w[stage-1])
            mask = stage_out_depth_crop>0
            error = 0*(dep_last_depth[mask]-stage_out_depth_crop[mask]).abs().mean()+ (gt_crop_depth[mask2]-stage_out_depth[mask2]).abs().mean()
        return stage_out,error
    
    def forward(self,features, blur_depth, gt, rgb, stage):
        outputs = {}
        #outputs["blur_disp"] = blur_depth
        #print(scale)
        all_scale = gt / blur_depth
        gt_origin = gt.clone()
        gt[all_scale>1.2] = 0
        gt[all_scale<0.8] = 0
        self.gt = gt
        gt_mask = gt.sign()
        blur_depth = gt_mask * gt + (1-gt_mask) * blur_depth
        rgbd = torch.cat((rgb, blur_depth),1)
        dep_0 = self.crop(gt,self.crop_h[0],self.crop_w[0])
        self.stage_block(features,rgbd,dep_0, stage, outputs)
        
        outputs["blur_disp"] = blur_depth
        outputs["disp_all_in"] = blur_depth
        
        outputs["dense_gt"] = self.crop(blur_depth,64,128)
        #outputs['scale'] = scale
        #outputs['scale'] = 1
        return outputs
class Iterative_3DPropagate(Iterative_Propagate):
    def __init__(self,crop_h,crop_w,mode='c',dropout=False):
        super().__init__(crop_h,crop_w,mode='c',dropout=False)
        self.model_ref0_3D = nn.Sequential(Conv3x3x3(4,4), Conv3x3x3(4,1))
        self.model_ref1_3D = nn.Sequential(Conv3x3x3(4,4), Conv3x3x3(4,1))
        self.model_ref2_3D = nn.Sequential(Conv3x3x3(4,4), Conv3x3x3(4,1))
        self.model_ref3_3D = nn.Sequential(Conv3x3x3(4,4), Conv3x3x3(4,1))
        self.model_ref4_3D = nn.Sequential(Conv3x3x3(4,4), Conv3x3x3(4,1))
        
        
        self.model_ref0 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref1 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref2 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref3 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,2),
                            ConvBlock(32,64,4),
                            ConvBlock(32,32,2),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref4 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,2),
                            ConvBlock(32,64,4),
                            ConvBlock(32,32,2),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.models_3D = nn.ModuleList([self.model_ref0_3D,self.model_ref1_3D,self.model_ref2_3D,self.model_ref3_3D,self.model_ref4_3D])
        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3,self.model_ref4])
    def make_3D_feature(self,feature,dep):
        median_value = torch.median(dep)
        mask_mlarge = dep > median_value
        mask_msmall = dep <= median_value
        #>
        median_4_3 = torch.median(dep[mask_mlarge])
        mask_4 = dep > median_4_3
        mask_3 = (dep <= median_4_3) * mask_mlarge
        #<
        median_4_1 = torch.median(dep[mask_msmall])
        mask_1 = dep <= median_4_1
        mask_2 = (dep > median_4_1) * mask_msmall

        mask = torch.cat((mask_1,mask_2,mask_3,mask_4),1)#B*4*H*W
        mask = mask.unsqueeze(2)#B*4*1*H*W
        
        m_4_3 = torch.median(dep[1-mask])
        feature_3D = feature.unsqueeze(1)
        feautre_3D = feature_3D.repeat(1,4,1,1,1)#B*4*C*H*W
        mask = mask.repeat(1,1,feature_3D.shape[1],1,1)#B*4*C*H*W
        feature_3D = feature_3D * mask
        return feautre_3D

    def stage_forward(self,features,rgbd,dep_last,stage):
        model = self.models[stage]
        model_3D = self.models_3D[stage]
        #dep_enc = self.dep_enc
        h = self.crop_h[stage]
        w = self.crop_w[stage]
        #dep_last is the padded depth
        rgbd = self.crop(rgbd,h,w)
        feature_crop = self.crop(features,h,w)
        dep_gt = self.crop(self.gt,h,w)
        dep = rgbd[:,3,:,:].unsqueeze(1)
        if dep[dep_last>0].shape != torch.Size([0]):
            if torch.median(dep[dep_last>0]) > 0:
                scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
            else:
                scale = 1
        else:
            scale = 1
            print("warning dep[dep_last>0] is empty,stage is %d"%stage)
        dep = dep * scale
        mask = dep_last.sign()
        mask_gt = dep_gt.sign()
        dep_fusion = dep_last * mask + dep * (1-mask)
        dep_fusion = dep_gt * mask_gt + dep_fusion * (1-mask_gt)
        feature_stage = torch.cat((feature_crop,dep_fusion),1)
        
        feature_3D = self.make_3D_feature(feature_stage,dep_fusion)
        feature_3D = model_3D(feature_3D)
        feature_3D = feature_3D.squeeze()

        feature_fusion = torch.cat((feature_3D,dep_fusion),1)

        dep = model(feature_fusion)
        return dep, feature_stage