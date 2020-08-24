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

class model_wrapper(nn.Module):
    def __init__(self,models,opt,device):
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
            self.refine_stage = list(range(options.refine_stage))
            if len(self.refine_stage) > 4:
                self.crop_h = [96,128,160,192,192]
                self.crop_w = [192,256,384,448,640]
            else:
                self.crop_h = [96,128,160,192]
                self.crop_w = [192,256,384,640]
        
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
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
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    if self.refine:
                        with torch.no_grad():
                            pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            if self.refine:
                with torch.no_grad():
                    axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    
                    pose_scale = 1 / outputs['scale']

                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i], invert=(f_i < 0), scale=pose_scale)

        return outputs
    
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]

            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                T = outputs[("cam_T_cam", 0, frame_id)]
                
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
        
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
        scales = self.opt.scales.copy()
        
        for scale in scales:
            loss = 0
            reprojection_losses = []

            source_scale = 0
            color = inputs[("color", 0, scale)]
            
            disp = outputs[("disp", scale)]
            target = inputs[("color", 0, source_scale)]
            disp_pred = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            depth_part_gt =  F.interpolate(
                    inputs["depth_gt_part"], [self.opt.height, self.opt.width], mode="nearest")
            disp_part_gt = depth_to_disp(depth_part_gt,self.opt.min_depth,self.opt.max_depth)
            mask = disp_part_gt>0
            depth_loss = torch.abs(disp_pred[mask] - disp_part_gt[mask]).mean()
            losses["loss/depth_{}".format(scale)] = depth_loss
            depth_loss = depth_loss
            if self.opt.refine:
                
                depth_loss_refine = torch.abs(disp_pred-outputs["dense_gt"]).mean()
                depth_loss += depth_loss_refine
                depth_loss *= 2

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
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
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def forward(self,inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        depth_part_gt =  F.interpolate(
                    inputs["depth_gt_part"], [self.opt.height, self.opt.width], mode="nearest")
        features = self.models["encoder"](torch.cat((inputs["color_aug", 0, 0],depth_part_gt),1))
        outputs = self.models["depth"](features)

        if self.opt.refine:
            with torch.no_grad():
                features_nograd = self.models["encoder_nograd"](inputs["color_aug", 0, 0])
                outputs_nograd = self.models["depth_nograd"](features_nograd)
                disp_blur = outputs_nograd[("disp", 0)]
                features = None
                inputs["depth_gt_part"] = F.interpolate(inputs["depth_gt_part"], [self.opt.height, self.opt.width], mode="nearest")
                disp_part_gt = depth_to_disp(inputs["depth_gt_part"] ,self.opt.min_depth,self.opt.max_depth)
                
                outputs_ref = self.models["mid_refine"](outputs_nograd["disp_feature"], disp_blur, disp_part_gt, inputs[("color_aug", 0, 0)],self.refine_stage)
                outputs["dense_gt"] = F.interpolate(outputs_ref[("disp",0)], [self.opt.height, self.opt.width], mode="bilinear")
                
                #outputs["disp_gt_part"] = disp_part_gt#after the forwar,the disp gt has been filtered
                #_,outputs["dense_gt"] = disp_to_depth(outputs["dense_gt"],self.opt.min_depth,self.opt.max_depth)
        
        if not (self.opt.use_stereo and self.opt.frame_ids == [0]):
            outputs.update(self.predict_poses(inputs))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses