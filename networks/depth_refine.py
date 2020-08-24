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
#from .deform_conv import DeformConv
class Simple_Propagate(nn.Module):
    def __init__(self,crop_h,crop_w,mode='c'):
        super(Simple_Propagate, self).__init__()
        self.crop_h = crop_h#[96,128,160,192]
        self.crop_w = crop_w#[192,256,384,640]
        self.model_ref0 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref1 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref2 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref3 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,64),
                            ConvBlock(64,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3])
        self.crop_mode = mode
    
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
    #small feature 256 -> fuse small gt -> conv -> larger area -> next stage 
    def stage_forward(self,features,rgbd,dep_last,stage):
        #dep_last is the padded depth
        model = self.models[stage]
        h = self.crop_h[stage]
        w = self.crop_w[stage]
        rgbd = self.crop(rgbd,h,w)
        feature_crop = self.crop(features,h,w)
        dep = rgbd[:,3,:,:].unsqueeze(1)
        mask = dep_last.sign()
        dep_fusion = dep_last * mask + dep * (1-mask) 
        feature_stage = torch.cat((feature_crop,dep_fusion),1)
        x = model(feature_stage)
        x, scale = self.scale_adjust(dep_last,x)
        return x, feature_stage
    
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
        blur_depth_o, scale = self.scale_adjust(gt,blur_depth)
        
        gt[all_scale>1.2] = 0
        gt[all_scale<0.8]=0
        gt_mask = gt.sign()
        blur_depth = gt_mask * gt + (1-gt_mask) * blur_depth_o
        rgbd = torch.cat((rgb, blur_depth_o),1)
        dep_0 = self.crop(gt,self.crop_h[0],self.crop_w[0])
        
        self.stage_block(features,rgbd,dep_0, stage, outputs)
        # if self.crop_mode == 'b':
        #     for ind,(h,w) in enumerate(zip(self.crop_h,self.crop_w)):
        #         outputs[("disp",ind)] = self.crop(blur_depth,h,w)
        
        outputs["blur_disp"] = blur_depth_o
        outputs["disp_all_in"] = blur_depth
        
        outputs["dense_gt"] = self.crop(blur_depth,64,128)
        outputs['scale'] = scale
        #outputs['scale'] = 1
        return outputs
class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class Depth_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(ConvBlock(1,4),
                            ConvBlock(4,8),
                            ConvBlock(8,16),
                            BasicBlock(16,16),
                            ConvBlock(16,32),
                            BasicBlock(32,32),
                            ConvBlock(32,16),
                            BasicBlock(16,16),
                            ConvBlock(16,4)
                            )
    def forward(self,x):
        x = self.model(x)
        return x

class Depth_encoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(ConvBlock(1,4),
                            ConvBlock(4,8),
                            ConvBlock(8,16),
                            ConvBlock(16,16),
                            ConvBlock(16,32),
                            ConvBlock(32,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            ConvBlock(8,4),
                            )
    def forward(self,x):
        x = self.model(x)
        return x
class Iterative_Propagate(Simple_Propagate):
    def __init__(self,crop_h,crop_w,mode,dropout=False):
        super(Iterative_Propagate, self).__init__(crop_h, crop_w,mode)
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
        dep = rgbd[:,3,:,:].unsqueeze(1)
        if torch.median(dep[dep_last>0]) > 0:
            scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
        else:
            scale = 1
        dep = dep * scale
        mask = dep_last.sign()
        for i in range(self.propagate_time): 
            dep_fusion = dep_last * mask + dep * (1-mask)
            feature_stage = torch.cat((feature_crop,dep_fusion),1)
            #feature_stage = F.dropout2d(feature_stage,0.2,training=self.dropout)
            dep = model(feature_stage)
            if torch.median(dep[dep_last>0]) > 0:
                scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
            else:
                scale = 1
            dep = dep * scale
        return dep, feature_stage



class Iterative_Propagate_old(Simple_Propagate):
    def __init__(self,crop_h,crop_w,mode,dropout=False):
        super().__init__(crop_h, crop_w,mode)
        self.model_ref0 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref1 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref2 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref3 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,2),
                            ConvBlock(32,16),
                            Conv3x3(16,1),nn.Sigmoid())
        self.model_ref4 = nn.Sequential(ConvBlock(16+1,32),
                            ConvBlock(32,32,4),
                            ConvBlock(32,16,2),
                            Conv3x3(16,1),nn.Sigmoid())

        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3])
        if len(self.crop_h) > 4:
            self.models.append(self.model_ref4)
        self.propagate_time = 1
        self.dropout = dropout
        
    
    def stage_forward(self,features,rgbd,dep_last,stage):
        model = self.models[stage]
        h = self.crop_h[stage]
        w = self.crop_w[stage]
        #dep_last is the padded depth
        rgbd = self.crop(rgbd,h,w)
        feature_crop = self.crop(features,h,w)
        #feature_crop = rgbd[:,1:,:,:]
        dep = rgbd[:,3,:,:].unsqueeze(1)
        if torch.median(dep[dep_last>0]) > 0:
            scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
        else:
            scale = 1
        dep = dep * scale
        mask = dep_last.sign()
        
        for i in range(self.propagate_time): 
            dep_fusion = dep_last * mask + dep * (1-mask)
            feature_stage = torch.cat((feature_crop,dep_fusion),1)
            feature_stage = F.dropout2d(feature_stage,0.2,training=self.dropout)
            dep = model(feature_stage)
            if torch.median(dep[dep_last>0]) > 0:
                scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
            else:
                scale = 1
            dep = dep * scale
        return dep, feature_stage

class Iterative_Propagate_seq(Simple_Propagate):
    def __init__(self,crop_h,crop_w,mode):
        super().__init__(crop_h, crop_w,mode)
        self.model_ref0 = nn.Sequential(ConvBlock(16+4,32),
                            ConvBlock(32,16),
                            ConvBlock(16,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref1 = nn.Sequential(ConvBlock(16+4,32),
                            ConvBlock(32,16),
                            ConvBlock(16,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref2 = nn.Sequential(ConvBlock(16+4,32),
                            ConvBlock(32,32),
                            ConvBlock(32,16),
                            ConvBlock(16,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref3 = nn.Sequential(ConvBlock(16+4,32),
                            ConvBlock(32,32,2),
                            ConvBlock(32,16),
                            ConvBlock(16,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())
        self.model_ref4 = nn.Sequential(ConvBlock(16+4,32),
                            ConvBlock(32,32,4),
                            ConvBlock(32,16,2),
                            ConvBlock(16,16),
                            ConvBlock(16,8),
                            Conv3x3(8,1),nn.Sigmoid())

        self.models = nn.ModuleList([self.model_ref0,self.model_ref1,self.model_ref2,self.model_ref3])
        self.models_seq = nn.ModuleList([Depth_encoder(),Depth_encoder(),Depth_encoder(),Depth_encoder()])
        if len(self.crop_h) > 4:
            self.models.append(self.model_ref4)
            self.models_seq.append(Depth_encoder())
        self.propagate_time = 1
        
    
    def stage_forward(self,features,rgbd,dep_last,stage):
        model = self.models[stage]
        model_seq = self.models_seq[stage]
        h = self.crop_h[stage]
        w = self.crop_w[stage]
        #dep_last is the padded depth
        rgbd = self.crop(rgbd,h,w)
        feature_crop = self.crop(features,h,w)
        #feature_crop = rgbd[:,1:,:,:]
        dep = rgbd[:,3,:,:].unsqueeze(1)
        if torch.median(dep[dep_last>0]) > 0:
            scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
        else:
            scale = 1
        dep = dep * scale
        mask = dep_last.sign()
        
        for i in range(self.propagate_time): 
            dep_fusion = dep_last * mask + dep * (1-mask)
            dep_fusion = model_seq(dep_fusion)
            feature_stage = torch.cat((feature_crop,dep_fusion),1)
            dep = model(feature_stage)
            if torch.median(dep[dep_last>0]) > 0:
                scale = torch.median(dep_last[dep_last>0]) / torch.median(dep[dep_last>0])
            else:
                scale = 1
            dep = dep * scale
        return dep, feature_stage



class Deformable_Conv(nn.Module):
    def __init__(self,inC,outC):
        super(Deformable_Conv,self).__init__()
        num_deformable_groups = 2
        kH, kW = 3,3
        self.offset_conv = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, (kH,kW),1,1,bias=False)
        self.deform_conv = DeformConv(inC, outC, (kH,kW), 1, 1, deformable_groups = num_deformable_groups)
        nn.init.constant_(self.offset_conv.weight, 0.)
        #nn.init.constant_(self.offset_conv.bias, 0.)
    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x,offset)
        return x
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=2, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale



class U_refine(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, pretrained=False, num_input_images=2):
        super(U_refine, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.num_output_channels = 1
        self.use_skips = True
        self.upsample_mode = 'nearest'
        self.scales = range(4)
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

         # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, blur_depth, gt, rgb):
        scale = torch.median(gt[gt>0]) / torch.median(blur_depth[gt>0])
        all_scale = gt / blur_depth
        blur_depth_o = blur_depth
        blur_depth_s = blur_depth * scale
        
        gt[all_scale>1.2] = 0
        gt[all_scale<0.8]=0
        gt_mask = gt.sign()
        blur_depth = gt_mask * gt + (1-gt_mask) * blur_depth_s
        x = rgb
        x = torch.cat((rgb,blur_depth,blur_depth,blur_depth),1)
        
        self.features = []
        x = self.encoder.conv1(x)
        
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        self.outputs = {}

        # decoder
        input_features = self.features
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                #mask = self.sigmoid(self.convs[("maskconv", i)](x))
        #depth_ref = self.outputs[("disp",0)]# * mask + (1-mask) * blur_depth_o
        depth_ref = self.outputs[("disp", 0)]
        scale_ref = torch.median(gt[gt>0]) / torch.median(depth_ref[gt>0])
        self.outputs["blur_disp"] = blur_depth_s
        self.outputs["disp_all_in"] = blur_depth
        self.outputs['scale'] = scale
        return self.outputs#, blur_depth, scale_ref








