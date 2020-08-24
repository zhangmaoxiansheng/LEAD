import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from layers import *

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential (
                    nn.Conv2d(4, 16, 3, 1, 1),
                    nn.BatchNorm2d(16),
                    nn.LeakyReLU(True),
                    nn.Conv2d(16, 8, 3, 2, 1), #/2
                    nn.BatchNorm2d(8),
                    nn.LeakyReLU(True),
                    nn.AvgPool2d(2),            #/2
                    nn.Conv2d(8, 4, 3, 2, 1), #/2
                    nn.BatchNorm2d(4),
                    nn.LeakyReLU(True),
                    nn.AvgPool2d(2),            #/2
                    nn.Conv2d(4, 1, 3, 1, 1), 
                    nn.LeakyReLU(True)
                )
    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator_group(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([Discriminator(), Discriminator(),Discriminator(),Discriminator(),Discriminator()])
    def forward(self,x,ind):
        x = self.models[ind](x)
        return x

class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        target_real_label = 1.0
        target_fake_label = 0.0
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

class pix2pix_loss(nn.Module):
    def __init__(self, opt_g, opt_d,netD, opt, mode='c'):
        super(pix2pix_loss, self).__init__()
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.netD = netD
        self.criterionGAN = GANLoss().cuda()
        
        self.min_depth = opt.min_depth
        self.max_depth = opt.max_depth
        self.refine_stage = list(range(opt.refine_stage))
        self.frame = [1]#opt.frame_ids[1:]

        self.stage_weight = [0.2,0.5,0.8,1]
        if len(self.refine_stage) > 4:
            #self.stage_weight = [0.2,0.2,0.5,0.8,1]
            self.stage_weight = [0.1,0.1,0.1,0.3,0.5]
        self.crop_mode = mode
        self.D_update = 0

    @staticmethod
    def set_requires_grad(net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def backward_G(self,inputs, outputs,losses):
        #step 1 fool the discriminator  try to make D(G(a)) -> 1
        #conditions: fused depth and RGB features
        #fake predout wapred RGB
        #GT           crop RGB
        GAN_loss_total = 0
        if self.epoch > self.start_gan and self.epoch < self.stop_gan:
            for s in self.refine_stage:
                GAN_loss_s = 0
                stage_weight_curr = self.stage_weight[s]
                for f_i in self.frame:
                    #condition = outputs[("condition",s)]
                    dep = outputs[("depth",0,s)]
                    rgb = outputs[("color",f_i,s)]
                    fake_rgb_cond = torch.cat((rgb,dep),1)
                    pred_fake = self.netD(fake_rgb_cond)
                    GAN_loss_s += self.criterionGAN(pred_fake,True) * stage_weight_curr
                GAN_loss_s = GAN_loss_s
                GAN_loss_total += GAN_loss_s
                losses["loss/G_{}".format(s)] = GAN_loss_total 
            losses["loss/G_total"] = GAN_loss_total / len(self.refine_stage)
            losses["loss"] += losses["loss/G_total"] * 0.01
        losses["loss"].backward()
        return losses
    
    def backward_D(self,inputs, outputs,losses):
        #Fake
        gt_shape = outputs[("dense_gt")].shape
        h = gt_shape[2]
        w = gt_shape[3]
        GAN_loss_fake = 0
        s = self.refine_stage[-1]
        f_i = 1
        GAN_loss_s = 0
        stage_weight_curr = self.stage_weight[s]
        #rand_scale = (2-0.5) * torch.rand(1).cuda() + 0.5
        rand_scale = (2-0.5) * np.random.rand(1) + 0.5
        rand_scale = rand_scale[0]
        max_scale_num = 1 / torch.max(outputs[("depth",0,s)])
        min_scale_num = 1 / torch.min(outputs[("depth",0,s)])
        rand_scale_num = (max_scale_num-min_scale_num) * torch.rand(1).cuda() + min_scale_num
        #for f_i in self.frame:
        fake_dep = self.crop(outputs[("depth",0,s)],h,w) * rand_scale_num
        fake_dep = F.interpolate(fake_dep, (int(h*rand_scale),int(w*rand_scale)))
        
        fake_rgb = self.crop(outputs[("color",f_i,s)],h,w)
        fake_rgb = F.interpolate(fake_rgb, (int(h*rand_scale),int(w*rand_scale)))

        fake_rgb_cond = torch.cat((fake_rgb,fake_dep),1)
        

        pred_fake = self.netD(fake_rgb_cond.detach())
        GAN_loss_s += self.criterionGAN(pred_fake,False) * stage_weight_curr
        GAN_loss_s = GAN_loss_s
        GAN_loss_fake += GAN_loss_s
        #real
        GAN_loss_real = 0
        GAN_loss_s = 0
        stage_weight_curr = self.stage_weight[s]
        #for f_i in self.frame:
        real_rgb = self.crop(inputs[("color",f_i,s)],h,w) * rand_scale_num
        real_dep = outputs[("dense_gt")]
        real_rgb_cond = F.interpolate(torch.cat((real_rgb,real_dep),1), (int(h*rand_scale), int(w*rand_scale)))
        pred_real = self.netD(real_rgb_cond.detach())
        GAN_loss_s += self.criterionGAN(pred_real,True) * stage_weight_curr
        GAN_loss_s = GAN_loss_s
        GAN_loss_real += GAN_loss_s
        GAN_loss_D = (GAN_loss_fake + GAN_loss_real) / 2
        losses["loss/D_total"] = GAN_loss_D
        GAN_loss_D.backward()
        return losses

    def crop(self,image,h=160,w=320):
        #mask = torch.zeros_like(image)
        origin_h = image.size(2)
        origin_w = image.size(3)
        if self.crop_mode=='c' or self.crop_mode=='s' or self.crop_mode=='r':
            h_start = max(int(round((origin_h-h)/2)),0)
            h_end = min(h_start + h,origin_h)
            w_start = max(int(round((origin_w-w)/2)),0)
            w_end = min(w_start + w,origin_w)
            output = image[:,:,h_start:h_end,w_start:w_end] 
        elif self.crop_mode=='b':
            origin_h = image.size(2)
            origin_w = image.size(3)
            h_start = max(int(round(origin_h-h)),0)
            w_start = max(int(round((origin_w-w)/2)),0)
            w_end = min(w_start + w,origin_w)
            output = image[:,:,h_start:,w_start:w_end] 
        
        return output
    
    def forward(self,inputs,outputs,losses,epoch):
        #D:
        self.epoch = epoch
        self.start_gan = 20#4
        self.stop_gan = 30
        outputs["D_update"] = False
        outputs["G_update"] = False
        #if epoch < 30:
        if epoch % 2 != 0 and epoch > self.start_gan and epoch < self.stop_gan:
            outputs["D_update"] = True
            outputs["G_update"] = False
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.opt_d.zero_grad()     # set D's gradients to zero
            losses.update(self.backward_D(inputs, outputs,losses))              # calculate gradients for D
            self.opt_d.step()
            self.D_update += 1          # update D's weights
            # else:
            #     outputs["D_update"] = False

        # update G
        else:
            if epoch > self.start_gan and epoch < self.stop_gan:
                outputs["G_update"] = True
            else:
                outputs["G_update"] = False
            outputs["D_update"] = False
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.opt_g.zero_grad()        # set G's gradients to zero
            losses.update(self.backward_G(inputs, outputs,losses))                   # calculate graidents for G
            self.opt_g.step()             # udpate G's weights


class pix2pix_loss_iter(pix2pix_loss):
    def __init__(self, opt_g, opt_d, netD, opt, crop_h, crop_w, mode='c'):
        super().__init__(opt_g, opt_d, netD, opt, mode=mode)
        self.crop_h = crop_h
        self.crop_w = crop_w
    
    
    def backward_D(self,inputs, outputs,losses):
        self.GAN_loss = 0
        #D_stage = list(range(len(self.crop_h)))
        
        for s in self.refine_stage:
            if s == 0:
                _,gt = disp_to_depth(outputs[("dense_gt")],self.min_depth,self.max_depth)
            else:
                gt = outputs[("depth",0,s-1)]
            h = gt.shape[2]
            w = gt.shape[3]
            gt_rgb = self.crop(inputs[("color",1,s)])
            gt_rgbd = torch.cat((gt_rgb,gt),1)

            fake_dep = self.crop(outputs[("depth",0,s)],h,w)
            fake_rgb = self.crop(outputs[("color",1,s)],h,w)
            fake_rgbd = torch.cat((fake_rgb,fake_dep),1)

            rand_scale = (1.2-0.8) * np.random.rand(1) + 0.8
            rand_scale = rand_scale[0]
            max_scale_num = 1.8
            min_scale_num = 0.5
            rand_scale_num = (max_scale_num-min_scale_num) * torch.rand(1).cuda() + min_scale_num

            gt_rgbd *= rand_scale_num
            fake_rgbd *= rand_scale_num
            gt_rgbd = F.interpolate(gt_rgbd, (int(h*rand_scale),int(w*rand_scale)))
            fake_rgbd = F.interpolate(fake_rgbd, (int(h*rand_scale),int(w*rand_scale)))

            stage_weight_curr = self.stage_weight[s]

            pred_fake = self.netD(fake_rgbd.detach())
            self.GAN_loss += self.criterionGAN(pred_fake,False) * stage_weight_curr
    
            pred_real = self.netD(gt_rgbd.detach())
            self.GAN_loss += self.criterionGAN(pred_real,True) * stage_weight_curr
        self.GAN_loss /= 2
        losses["loss/D_total"] = self.GAN_loss
        self.GAN_loss.backward()
        return losses

class pix2pix_loss_iter2(pix2pix_loss):
    def __init__(self, opt_g, opt_d, netD, opt, crop_h, crop_w, mode='c'):
        super().__init__(opt_g, opt_d, netD, opt, mode=mode)
        self.crop_h = crop_h
        self.crop_w = crop_w
    
    
    def backward_D(self,inputs, outputs,losses):
        self.GAN_loss = 0
        #D_stage = list(range(len(self.crop_h)))
        
        for s in self.refine_stage:
            if s == 0:
                _,gt = disp_to_depth(outputs[("dense_gt")],self.min_depth,self.max_depth)
            else:
                gt = outputs[("depth",0,s-1)]
            h = gt.shape[2]
            w = gt.shape[3]
            gt_rgb = self.crop(inputs[("color",1,s)],h,w)
            gt_rgbd = torch.cat((gt_rgb,gt),1)

            fake_dep = self.crop(outputs[("depth",0,s)],h,w)
            fake_rgb = self.crop(outputs[("color",1,s)],h,w)
            fake_rgbd = torch.cat((fake_rgb,fake_dep),1)

            rand_scale = (1.2-0.8) * np.random.rand(1) + 0.8
            rand_scale = rand_scale[0]
            max_scale_num = 1.8
            min_scale_num = 0.5
            rand_scale_num = (max_scale_num-min_scale_num) * torch.rand(1).cuda() + min_scale_num

            gt_rgbd *= rand_scale_num
            fake_rgbd *= rand_scale_num
            gt_rgbd = F.interpolate(gt_rgbd, (int(h*rand_scale),int(w*rand_scale)))
            fake_rgbd = F.interpolate(fake_rgbd, (int(h*rand_scale),int(w*rand_scale)))

            stage_weight_curr = self.stage_weight[s]

            pred_fake = self.netD(fake_rgbd.detach(),s)
            self.GAN_loss += self.criterionGAN(pred_fake,False) * stage_weight_curr
    
            pred_real = self.netD(gt_rgbd.detach(),s)
            self.GAN_loss += self.criterionGAN(pred_real,True) * stage_weight_curr
        self.GAN_loss /= 2
        losses["loss/D_total"] = self.GAN_loss
        self.GAN_loss.backward()
        return losses

    def backward_G(self,inputs, outputs,losses):
        #step 1 fool the discriminator  try to make D(G(a)) -> 1
        #conditions: fused depth and RGB features
        #fake predout wapred RGB
        #GT           crop RGB
        GAN_loss_total = 0
        if self.epoch > self.start_gan and self.epoch < self.stop_gan:
            for s in self.refine_stage:
                GAN_loss_s = 0
                stage_weight_curr = self.stage_weight[s]
                for f_i in self.frame:
                    #condition = outputs[("condition",s)]
                    dep = outputs[("depth",0,s)]
                    rgb = outputs[("color",f_i,s)]
                    fake_rgb_cond = torch.cat((rgb,dep),1)
                    pred_fake = self.netD(fake_rgb_cond,s)
                    GAN_loss_s += self.criterionGAN(pred_fake,True) * stage_weight_curr
                GAN_loss_s = GAN_loss_s
                GAN_loss_total += GAN_loss_s
                losses["loss/G_{}".format(s)] = GAN_loss_total 
            losses["loss/G_total"] = GAN_loss_total / len(self.refine_stage)
            losses["loss"] += losses["loss/G_total"] * 0.05
        losses["loss"].backward()
        return losses


