# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

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
#from IPython import embed
import cv2
import copy
cv2.setNumThreads(0)


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.dropout = options.dropout
        self.refine_stage = list(range(options.refine_stage))
        self.refine = options.refine
        self.crop_mode = options.crop_mode
        self.gan = options.gan
        self.gan2 = options.gan2
        self.edge_refine = options.edge_refine
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.parameters_to_train_refine = []


        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        if self.refine:
            if len(self.refine_stage) > 4:
                if self.crop_mode == 'b' or self.crop_mode == 'cl':
                    self.crop_h = [128,168,192,192,192]
                    self.crop_w = [192,256,384,448,640]
                    # self.crop_h = [192,192,192,192,192]
                    # self.crop_w = [192,256,384,448,640]
                else:
                    self.crop_h = [96,128,160,192,192]
                    self.crop_w = [192,256,384,448,640]
            else:
                self.crop_h = [96,128,160,192]
                self.crop_w = [192,256,384,640]
            if self.opt.refine_model == 's':
                self.models["mid_refine"] = networks.Simple_Propagate(self.crop_h,self.crop_w,self.crop_mode,self.dropout)
            elif self.opt.refine_model == 'i':
                self.models["mid_refine"] = networks.Iterative_Propagate(self.crop_h,self.crop_w,self.crop_mode,False)
            elif self.opt.refine_model == 'is':
                self.models["mid_refine"] = networks.Iterative_Propagate_seq(self.crop_h,self.crop_w,self.crop_mode,False)
            elif self.opt.refine_model == 'io':
                self.models["mid_refine"] = networks.Iterative_Propagate_old(self.crop_h,self.crop_w,self.crop_mode,False)

            self.models["mid_refine"].to(self.device)
            self.parameters_to_train_refine += list(self.models["mid_refine"].parameters())
            if self.gan:
                self.models["netD"] = networks.Discriminator()
                self.models["netD"].to(self.device)
                self.parameters_D = list(self.models["netD"].parameters())
            if self.gan2:
                self.models["netD"] = networks.Discriminator_group()
                self.models["netD"].to(self.device)
                self.parameters_D = list(self.models["netD"].parameters())

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=1)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        if self.refine:
            for param in self.models["encoder"].parameters():
                param.requeires_grad = False

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales,refine=self.refine)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["pose_encoder"] = networks.ResnetEncoder(self.opt.num_layers,self.opt.weights_init == "pretrained",num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        if self.refine:
            for param in self.models["pose_encoder"].parameters():
                param.requeires_grad = False

        self.models["pose"] = networks.PoseDecoder(self.models["pose_encoder"].num_ch_enc,num_input_features=1,num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())
        if self.refine:
            for param in self.models["pose"].parameters():
                param.requeires_grad = False

        

        if self.opt.load_weights_folder is not None:
            self.load_model()
        if self.refine:
            self.models["depth_ref"] = copy.deepcopy(self.models["depth"])
            self.models["depth_ref"].to(self.device)
            #self.parameters_to_train_refine += list(self.models["depth_ref"].parameters())
            for param in self.models["depth"].parameters():
                param.requeires_grad = False
            for param in self.models["depth_ref"].parameters():
                param.requeires_grad = False
        if self.refine:
            parameters_to_train = self.parameters_to_train_refine
        else:
            parameters_to_train = self.parameters_to_train      

        
        self.model_optimizer = optim.Adam(parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.gan or self.gan2:
            self.D_optimizer = optim.Adam(self.parameters_D, 1e-4)
            self.model_lr_scheduler_D = optim.lr_scheduler.StepLR(
            self.D_optimizer, self.opt.scheduler_step_size, 0.1)
            if self.gan:
                self.pix2pix = networks.pix2pix_loss_iter(self.model_optimizer, self.D_optimizer, self.models["netD"], self.opt, self.crop_h, self.crop_w, mode=self.crop_mode,)
            else:
                self.pix2pix = networks.pix2pix_loss_iter2(self.model_optimizer, self.D_optimizer, self.models["netD"], self.opt, self.crop_h, self.crop_w, mode=self.crop_mode,)
        
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "kitti_depth":datasets.KITTIDepthDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files_p.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, refine=self.refine, crop_mode=self.crop_mode, crop_h=self.crop_h, crop_w=self.crop_w)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, refine=self.refine, crop_mode=self.crop_mode, crop_h=self.crop_h, crop_w=self.crop_w)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.refine_stage:
            h = self.crop_h[scale]
            w = self.crop_w[scale]

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        if self.opt.load_weights_folder is None:
            self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.epoch, self.epoch + self.opt.num_epochs):
            self.run_epoch()
            if self.epoch>10 and (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)
            
            if self.gan or self.gan2:
                self.pix2pix(inputs, outputs, losses, self.epoch)
            else:
                self.model_optimizer.zero_grad()
                losses["loss"].backward()
                self.model_optimizer.step()

            duration = time.time() - before_op_time
            
            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 4000
            late_phase = self.step % 2000 == 0

            if (self.gan or self.gan2) and batch_idx % self.opt.log_frequency :
                if outputs["D_update"]:
                    self.log_time(batch_idx, duration, losses["loss/D_total"].cpu().data)
                if outputs["G_update"]:
                    self.log_time(batch_idx, duration, losses["loss/G_total"].cpu().data)
            self.log_time(batch_idx, duration, losses["loss"].cpu().data)

            if early_phase or late_phase:
                #self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                #self.val()
            self.step += 1
        self.model_lr_scheduler.step()
            

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
        inputs["depth_gt_part"] =  F.interpolate(inputs["depth_gt_part"], [self.opt.height, self.opt.width], mode="nearest")
        with torch.no_grad():
            features = self.models["encoder"](torch.cat((inputs["color_aug", 0, 0],inputs["depth_gt_part"]),1))
            outputs = self.models["depth"](features)

        if self.refine:
            disp_blur = outputs[("disp", 0)]
            disp_part_gt = depth_to_disp(inputs["depth_gt_part"] ,self.opt.min_depth,self.opt.max_depth)
            if (self.gan or self.gan2) and self.epoch % 2 != 0 and self.epoch > self.pix2pix.start_gan and self.epoch < self.pix2pix.stop_gan:
                with torch.no_grad():
                    outputs.update(self.models["depth_ref"](features,self.dropout))
                    outputs.update(self.models["mid_refine"](outputs["disp_feature"], disp_blur, disp_part_gt, inputs[("color_aug", 0, 0)],self.refine_stage))
            else:
                with torch.no_grad():
                    outputs.update(self.models["depth_ref"](features,self.dropout))
                outputs.update(self.models["mid_refine"](outputs["disp_feature"], disp_blur, disp_part_gt, inputs[("color_aug", 0, 0)],self.refine_stage))
            outputs["disp_gt_part"] = disp_part_gt#after the forwar,the disp gt has been filtered
            _,outputs["dense_gt"] = disp_to_depth(outputs["dense_gt"],self.opt.min_depth,self.opt.max_depth)
        

        for i in self.opt.frame_ids:
            origin_color = inputs[("color",i,0)].clone()
            for s in self.refine_stage:
                inputs[("color",i,s)] = self.crop(origin_color,self.crop_h[s],self.crop_w[s])
        if not self.opt.surp_depth:
            if self.use_pose_net:
                outputs.update(self.predict_poses(inputs, features))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        if self.refine:
            losses["loss/scale"] = outputs["scale"]

        return outputs, losses

    def predict_poses(self, inputs, features):
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

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()
    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        
        for scale in self.refine_stage:
            disp = outputs[("disp", scale)]

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth
            if not self.opt.surp_depth:
                for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                    T = outputs[("cam_T_cam", 0, frame_id)]

                    cam_points = self.backproject_depth[scale](
                        depth, inputs[("inv_K", scale)])
                    pix_coords = self.project_3d[scale](
                        cam_points, inputs[("K", scale)], T)

                    outputs[("sample", frame_id, scale)] = pix_coords

                    outputs[("color", frame_id, scale)] = F.grid_sample(
                        inputs[("color", frame_id, scale)],
                        outputs[("sample", frame_id, scale)],
                        padding_mode="border")

                    if not self.opt.disable_automasking:
                        outputs[("color_identity", frame_id, scale)] = \
                            inputs[("color", frame_id, scale)]

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
        stage_weight = [1,1,1.5,2]
        if len(self.refine_stage) > 4:
            #stage_weight = [1,1,1.2,1.2,2]
            stage_weight = [1,1,1.2,1.2,1.5]
            #stage_weight = [0.25,0.5,0.8,1,1.5]

       
        for scale in self.refine_stage:
        
            h = self.crop_h[scale]
            w = self.crop_w[scale]
            
            loss = 0
            reprojection_losses = []

            source_scale = 0
            color = inputs[("color", 0, scale)]
            disp = outputs[("disp", scale)]
            disp_pred = disp
            warning = torch.sum(torch.isnan(disp))
            if warning:
                print("nan in disppred")
            disp_target = self.crop(outputs["blur_disp"],h,w)

            target = inputs[("color", 0, scale)]
            depth_pred = outputs[("depth", 0, scale)]

            disp_part_gt = self.crop(outputs["disp_gt_part"],h,w)
            mask = disp_part_gt > 0
            depth_loss = torch.abs(disp_pred[mask] - disp_part_gt[mask]).mean()
            losses["loss/depth_{}".format(scale)] = depth_loss
            depth_loss = depth_loss
            if self.refine:   
                depth_l1_loss = torch.mean((disp - disp_target).abs())
                depth_ssim_loss = self.ssim(disp, disp_target).mean()
                if not self.opt.surp_depth:
                    depth_loss += depth_ssim_loss * 0.85 + depth_l1_loss * 0.15
                else:
                    depth_loss += depth_l1_loss * 10
                #depth_loss += depth_ssim_loss * 0.85 + depth_l1_loss * 0.15
                losses["loss/depth_ssim{}".format(scale)] = depth_ssim_loss
            if not self.opt.surp_depth:
                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                reprojection_losses = torch.cat(reprojection_losses, 1)

                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color", frame_id, scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # save both images, and do min all at once below
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
            if self.edge_refine and scale == self.refine_stage[-1]:
                mask_edge = F.interpolate(inputs["mask_edge"],(self.opt.height,self.opt.width))
                mask_edge_x2 = mask_edge[:,:,:,:-2]
                mask_edge_y2 = mask_edge[:,:,:-2,:]
                smooth_loss_edge = torch.mean(grad_disp_x2[mask_edge_x2>0]) + torch.mean(grad_disp_y2[mask_edge_y2>0])
                loss += self.opt.disparity_smoothness * smooth_loss_edge * 5
            
            # if self.refine_stage:
            #     loss += self.opt.disparity_smoothness * smooth_loss
            # else:
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                
            loss += depth_loss

            total_loss += loss * stage_weight[scale]
            losses["loss/{}".format(scale)] = loss
        total_loss /= len(self.refine_stage)
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        if self.refine:
            s = self.refine_stage[-1]
            depth_pred = outputs[("depth", 0, s)]
        else:
            depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        #depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)  //fuck!

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        scales_ = self.refine_stage
        # if self.refine:
        #     scales_.append('r')
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in scales_:
                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
                to_save['epoch'] = self.epoch
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            if os.path.exists(path):
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                if 'epoch' in pretrained_dict.keys():
                    self.epoch = pretrained_dict['epoch']
                else:
                    self.epoch = 0
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        # loading adam state
        # optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        # if os.path.isfile(optimizer_load_path):
        #     print("Loading Adam weights")
        #     optimizer_dict = torch.load(optimizer_load_path)
        #     self.model_optimizer.load_state_dict(optimizer_dict)
        # else:
        #     print("Cannot find Adam weights so Adam is randomly initialized")
    
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
