from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
import scipy.io as scio 
import copy
import pdb

class model_wrapper(nn.Module):
    def __init__(self,models,opt,device,crop_h,crop_w):
        super().__init__()
        self.models = models
        self.opt = opt
        assert self.opt.height % 32 == 0
        assert self.opt.width % 32 == 0
        self.device = device

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        if self.opt.refine:
            self.crop_mode = opt.crop_mode
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            self.crop_h = None
            self.crop_w = None
        
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        self.stage_weight = [1,1,1.2,1.2,1.5]

        self.backproject_depth = {}
        self.project_3d = {}
        if self.opt.refine:
            self.refine_stage = list(range(opt.refine_stage))
            scales = self.refine_stage
        else:
            scales = self.opt.scales
        for scale in scales:
            if self.opt.refine:
                h = self.crop_h[scale]
                w = self.crop_w[scale]
            else:
                h = self.opt.height // (2 ** scale)
                w = self.opt.width // (2 ** scale)
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
    
    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                axisangle, translation = self.models["pose"](pose_inputs)
                
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            pose_inputs = torch.cat(
                [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

            if self.opt.refine:
                with torch.no_grad():
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]
                    axisangle, translation = self.models["pose"](pose_inputs)
            else:
                pose_inputs = [self.models["pose_encoder"](pose_inputs)]
                axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i], invert=(f_i < 0))
        return outputs
    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        scales = self.refine_stage if self.opt.refine else self.opt.scales
        for scale in scales:
            disp = outputs[("disp", scale)]

            if not self.opt.refine:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                wrap_scale = scale if self.opt.refine else source_scale
                k_key = 'K_r' if self.opt.refine else 'K'
                inv_k_key = 'inv_K_r' if self.opt.refine else 'inv_K'
                T = outputs[("cam_T_cam", 0, frame_id)]
                
                cam_points = self.backproject_depth[wrap_scale](
                    depth, inputs[(inv_k_key, wrap_scale)])
                pix_coords = self.project_3d[wrap_scale](
                    cam_points, inputs[(k_key, wrap_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, wrap_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    scale_id = scale if self.opt.refine else source_scale
                    outputs[("color_identity", frame_id, scale_id)] = \
                        inputs[("color", frame_id, scale_id)]
        
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        scales =  self.refine_stage.copy() if self.opt.refine else self.opt.scales.copy()
        
        for scale in scales:
            loss = 0
            depth_loss = 0
            reprojection_losses = []

            source_scale = 0
            color = inputs[("color", 0, scale)]
            
            disp = outputs[("disp", scale)]
            if self.opt.refine:
                h = self.crop_h[scale]
                w = self.crop_w[scale]
                warning = torch.sum(torch.isnan(disp))
                if warning:
                    print("nan in disp")
                disp_pred = disp
                disp_target = self.crop(outputs["blur_disp"],h,w)
                target = inputs[("color", 0, scale)]
                depth_pred = outputs[("depth",0,scale)]
                disp_part_gt = self.crop(outputs["disp_gt_part"],h,w)
                depth_l1_loss = torch.mean((disp - disp_target).abs())
                depth_ssim_loss = self.ssim(disp, disp_target).mean()
                #depth_loss = depth_l1_loss * 0.25
                depth_loss += depth_ssim_loss*0.2 #depth_ssim_loss * 0.85 + depth_l1_loss * 0.15
                losses["loss/depth_ssim{}".format(scale)] = depth_ssim_loss
            else:
                target = inputs[("color", 0, source_scale)]
                disp_pred = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                disp_part_gt = depth_to_disp(inputs["depth_gt_part"],self.opt.min_depth,self.opt.max_depth)
            
            mask = disp_part_gt>0
            depth_loss += torch.abs(disp_pred[mask] - disp_part_gt[mask]).mean()*3
            losses["loss/depth_{}".format(scale)] = depth_loss

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                scale_id = scale if self.opt.refine else source_scale
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, scale_id)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                identity_reprojection_loss = identity_reprojection_losses

            
            reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()
            loss += to_optimise.mean()
            #losses["optloss/{}".format(scale)] = to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            grad_disp_x, grad_disp_y, grad_disp_x2, grad_disp_y2 = get_smooth_loss(norm_disp, color)
            smooth_loss = grad_disp_x.mean() + grad_disp_y.mean()
            
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            loss += depth_loss
            #special loss for our data
            if (self.opt.refine and scale == sclaes[-1]) or (not self.opt.refine and scale == 0):
                smooth_loss_aug = torch.mean(grad_disp_y[:,:,int(0.9*self.opt.height):,:]) * self.opt.disparity_smoothness * 5
                loss += smooth_loss_aug

            total_loss += loss * self.stage_weight[scale] if self.opt.refine else loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= len(scales)
        losses["loss"] = total_loss
        return losses

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
    
    def forward(self,inputs,epoch):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        inputs["depth_gt_part"] =  F.interpolate(
                    inputs["depth_gt_part"], [self.opt.height, self.opt.width], mode="nearest")
        if self.opt.refine:
            with torch.no_grad():
                features = self.models["encoder"](torch.cat((inputs["color_aug", 0, 0],inputs["depth_gt_part"]),1))
                outputs = self.models["depth"](features)
            disp_blur = outputs[("disp",0)]
            disp_part_gt = depth_to_disp(inputs["depth_gt_part"] ,self.opt.min_depth,self.opt.max_depth)
            #pdb.set_trace()
            with torch.no_grad():
                outputs.update(self.models["depth"](features,self.opt.dropout))
            if (self.opt.gan or self.opt.gan2) and epoch % 2 != 0 and epoch > self.opt.start_gan and epoch < self.opt.stop_gan:
                with torch.no_grad():
                    outputs.update(self.models["mid_refine"](outputs["disp_feature"], disp_blur, disp_part_gt, inputs[("color_aug", 0, 0)],self.refine_stage))
            else:
                outputs.update(self.models["mid_refine"](outputs["disp_feature"], disp_blur, disp_part_gt, inputs[("color_aug", 0, 0)],self.refine_stage))
            outputs["disp_gt_part"] = disp_part_gt#after the forwar,the disp gt has been filtered
            _,outputs["dense_gt"] = disp_to_depth(outputs["dense_gt"],self.opt.min_depth,self.opt.max_depth)
            for i in self.opt.frame_ids:
                origin_color = inputs[("color",i,0)].clone()
                for s in self.refine_stage:
                    inputs[("color",i,s)] = self.crop(origin_color,self.crop_h[s],self.crop_w[s])
        else:
            features = self.models["encoder"](torch.cat((inputs["color_aug", 0, 0],inputs["depth_gt_part"]),1))
            outputs = self.models["depth"](features)
        
        if not (self.opt.use_stereo and self.opt.frame_ids == [0]):
            outputs.update(self.predict_poses(inputs))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses