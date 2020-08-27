
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
from model_wrapper import model_wrapper
from model_joint_wrapper import model_joint_wrapper
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import pdb
#from apex import amp
torch.distributed.init_process_group(backend="nccl",init_method='env://')
#from IPython import embed
def set_requeires_grad(model,grad=False):
    for param in model.parameters():
        param.requeires_grad=grad
class Trainer:
    def __init__(self, options):
        self.opt = options
        self.refine = options.refine or options.inv_refine
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        self.crop_mode = options.crop_mode

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = self.opt.pose_model_input
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        self.models = {}
        self.parameters_to_train = []

        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        if self.opt.refine:
            if self.crop_mode == 'b' or self.crop_mode == 'cl':
                self.crop_h = [128,168,192,192,192]
                self.crop_w = [192,256,384,448,640]
            else:
                self.crop_h = [96,128,160,192,192]
                self.crop_w = [192,256,384,448,640]
        else:
            self.crop_h = None
            self.crop_w = None

        if self.refine:
            self.models["mid_refine"] = networks.Iterative_Propagate(self.crop_h,self.crop_w,self.crop_mode,False)

            self.models["mid_refine"].to(self.device)
            self.parameters_to_train += list(self.models["mid_refine"].parameters())
            if self.opt.gan:
                self.models["netD"] = networks.Discriminator()
                self.models["netD"].to(self.device)
                self.parameters_D = list(self.models["netD"].parameters())
            if self.opt.gan2:
                self.models["netD"] = networks.Discriminator_group()
                self.models["netD"].to(self.device)
                self.parameters_D = list(self.models["netD"].parameters())
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images=1)
        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales,refine=self.refine)
        self.models["depth"].to(self.device)

        if self.use_pose_net:
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["pose_encoder"].to(self.device)
            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            self.models["pose"].to(self.device)
            
        if self.refine and not self.opt.join:
            set_requeires_grad(self.models["depth"])
            set_requeires_grad(self.models["encoder"])
            set_requeires_grad(self.models["pose_encoder"])
            set_requeires_grad(self.models["pose"])
        elif self.opt.join:
            set_requeires_grad(self.models["pose_encoder"])
            set_requeires_grad(self.models["pose"])
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
        else:
            self.parameters_to_train += list(self.models["encoder"].parameters())
            self.parameters_to_train += list(self.models["depth"].parameters())
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
            self.parameters_to_train += list(self.models["pose"].parameters())

        parameters_to_train = self.parameters_to_train
        self.model_optimizer = optim.Adam(parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)
        if self.opt.gan or self.opt.gan2:
            self.D_optimizer = optim.Adam(self.parameters_D, 1e-4)
            self.model_lr_scheduler_D = optim.lr_scheduler.StepLR(self.D_optimizer, self.opt.scheduler_step_size, 0.1)
            if self.opt.gan:
                self.pix2pix = networks.pix2pix_loss_iter(self.model_optimizer, self.D_optimizer, self.models["netD"], self.opt, self.crop_h, self.crop_w, mode=self.crop_mode,)
            else:
                self.pix2pix = networks.pix2pix_loss_iter2(self.model_optimizer, self.D_optimizer, self.models["netD"], self.opt, self.crop_h, self.crop_w, mode=self.crop_mode,)
        if self.opt.load_weights_folder is not None:
            self.load_model()

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
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, refine=self.opt.refine, crop_mode=self.crop_mode,crop_h=self.crop_h, crop_w=self.crop_w)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,sampler=DistributedSampler(train_dataset))
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, refine=self.opt.refine, crop_mode=self.crop_mode,crop_h=self.crop_h, crop_w=self.crop_w)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True,sampler=DistributedSampler(val_dataset))
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()
        if self.opt.join:
            self.model_wrapper = model_joint_wrapper(self.models,self.opt,self.device)
        else:
            self.model_wrapper = model_wrapper(self.models,self.opt,self.device)
        self.model_wrapper.to(self.device)
        #amp.initialize(list(self.models.values()),self.model_optimizer,opt_level="O1")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            for key in self.models.keys():
                self.models[key] = torch.nn.parallel.DistributedDataParallel(self.models[key],
                                                      device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False,find_unused_parameters=True)

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
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if self.epoch > 10 and (self.epoch + 1) % self.opt.save_frequency == 0 and torch.distributed.get_rank()==0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.set_train()
        tbar = tqdm(self.train_loader)

        for batch_idx, inputs in enumerate(tbar):

            before_op_time = time.time()

            outputs, losses = self.model_wrapper.forward(inputs,self.epoch)

            if self.opt.gan or self.opt.gan2:
                self.pix2pix(inputs, outputs, losses, self.epoch)
            else:
                self.model_optimizer.zero_grad()
                # with amp.scale_loss(losses["loss"],self.model_optimizer) as scaled_loss:
                #     scaled_loss.backward()
                #losses["loss"] = losses["loss"].mean()
                losses["loss"].backward()
                self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if (self.opt.gan or self.opt.gan2):
                if outputs["D_update"]:
                    tbar.set_description("epoch {:>3} | batch {:>6} | loss: {:.5f} | D_loss: {:.5f}".format(self.epoch, batch_idx, losses["loss"].cpu().data,losses["loss/D_total"].cpu().data))
                elif outputs["G_update"]:
                    tbar.set_description("epoch {:>3} | batch {:>6} | loss: {:.5f} | G_loss: {:.5f}".format(self.epoch, batch_idx, losses["loss"].cpu().data,losses["loss/G_total"].cpu().data))
                else:
                    tbar.set_description("epoch {:>3} | batch {:>6} | loss: {:.5f}".format(self.epoch, batch_idx, losses["loss"].cpu().data))
            else:
                tbar.set_description("epoch {:>3} | batch {:>6} | loss: {:.5f}".format(self.epoch, batch_idx, losses["loss"].cpu().data))
            if early_phase or late_phase:
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                #self.val()

            self.step += 1
        self.model_lr_scheduler.step()

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
            outputs, losses = self.model_wrapper.forward(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        if self.refine:
            depth_pred = outputs[("depth", 0, 0)]
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
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

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
        scales_ = self.opt.scales.copy()
        # if self.refine:
        #     scales_.append('r')
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in scales_:
                for frame_id in self.opt.frame_ids:
                    if s != 'r':
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)
                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir,exist_ok=True)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        os.makedirs(save_folder,exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
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
            
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            if os.path.exists(path):
                print("Loading {} weights...".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                if 'epoch' in pretrained_dict.keys():
                    self.epoch = pretrained_dict['epoch']
                else:
                    self.epoch = 0
                pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() if k.replace('module.','') in model_dict}
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
