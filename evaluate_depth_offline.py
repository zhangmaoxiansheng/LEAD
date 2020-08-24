from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth, depth_to_disp
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import torch.nn.functional as F

from matplotlib import pyplot as plt

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def uncertainty_map(res_path):
    uncertainty_dir = os.path.join(res_path,'uncertainty_map')

    if not os.path.exists(uncertainty_dir):
        os.mkdir(uncertainty_dir)
    filenames = readlines(os.path.join(splits_dir, "eigen_benchmark", "test_files.txt"))
    for i in range(len(filenames)):
        imageset = np.load(os.path.join(res_path,'{}_stage4.npy'.format(i)))
        uncert = np.std(imageset,0)
        plt.imsave(os.path.join(uncertainty_dir,'{}_uncert.png'.format(i)),uncert,cmap='Greys')

def crop_center(image,h=160,w=320):
    origin_h = image.shape[0]
    origin_w = image.shape[1]
    h_start = max(int(round((origin_h-h)/2)),0)
    h_end = min(h_start + h,origin_h)
    w_start = max(int(round((origin_w-w)/2)),0)
    w_end = min(w_start + w,origin_w)
    output = image[h_start:h_end,w_start:w_end] 
    return output



def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def l1_loss(img1,gt):
    mask = gt>0
    return np.mean(np.abs(img1[mask] - gt[mask]))

def abs_rel(img1,gt):
    mask = gt>0
    return np.mean(np.abs(img1[mask] - gt[mask])/gt[mask])


def l2_loss(img1,gt):
    mask = gt>0
    return np.sqrt(np.mean((img1[mask] - gt[mask])**2))

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    opt.refine_stage = 5
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    res_base_path = opt.eval_out_dir
    
    crop_h = [96,128,160,192,192]
    crop_w = [192,256,384,448,640]

    filenames = readlines(os.path.join(splits_dir, "eigen_benchmark", "test_files.txt"))
    pred_disps = []
    gt = np.load(os.path.join(res_base_path,'gt.npy'))
    part_gt = np.load(os.path.join(res_base_path,'part_gt.npy'))
    saved_output = {}
    for s in range(opt.refine_stage):
        saved_output[s] = []
    for i in range(len(filenames)):
        print(i)
        for s in range(opt.refine_stage):
            img_name = os.path.join(res_base_path,'{}_stage{}.npy'.format(i,s))
            img = np.load(img_name)
            saved_output[s].append(img)

    #mean
    # for imageset in saved_output[4]:
    #     disp_mean = np.expand_dims(np.mean(imageset,0),0)
    #     scale_disp_mean,_ = disp_to_depth(disp_mean,opt.min_depth,opt.max_depth)
    #     pred_disps.append(scale_disp_mean)
    #l1/l2 + gt
    loss = l1_loss
    loss2 = abs_rel
    error_offline = []
    iter_time = opt.iter_time
    for i in range(len(filenames)):
        stage_best_depth = []
        print(i)
        for s in range(opt.refine_stage):
            print(s)
            imageset = saved_output[s][i]
            part_gt_now = part_gt[i]
            part_gt_now_stage = np.expand_dims(crop_center(part_gt_now,crop_h[s],crop_w[s]),0)#depth
            #part_gt_now_stage = depth_to_disp(part_gt_now_stage,opt.min_depth,opt.max_depth)
            images = np.split(imageset,iter_time,0)
            for j in range(iter_time):#iter times
                _,img_current = disp_to_depth(images[j],opt.min_depth,opt.max_depth)#depth
                #img_current = images[j]
                if s == 0:
                    error = loss(img_current,part_gt_now_stage)
                else:
                    dep_last = stage_best_depth[s-1]
                    img_current_last = np.expand_dims(crop_center(np.squeeze(img_current),crop_h[s-1],crop_w[s-1]),0)            
                    error = loss(img_current,part_gt_now_stage) + loss(img_current_last,dep_last)# + abs_rel(img_current_last,dep_last)
                error_offline.append(error)
                if j == 0:
                    best_error = error
                    best_img = img_current
                elif error < best_error:
                    best_error = error
                    best_img = img_current
            stage_best_depth.append(best_img)  
        #final_disp = stage_best_depth[-1]
        #scale_disp,_ = disp_to_depth(final_disp,opt.min_depth,opt.max_depth)
        pred_disps.append(stage_best_depth[-1])         
                
    pred_disps = np.concatenate(pred_disps,0)
    print(pred_disps.shape)

    gt_depths = gt
    print("-> Evaluating")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_depth = cv2.resize(gt_depth, (1242, 375))
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        #pred_depth = 1 / pred_disp
        pred_depth = pred_disp
        mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)

    if opt.save_pred_disps:
        np.save(os.path.join(res_base_path,'offline_depres.npy'),pred_disps)
        np.save(os.path.join(res_base_path,'error_offline.npy'),error_offline)
    
    line1 = "\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
    print(line1)
    with open(os.path.join(res_base_path,'res.txt'),'a') as f:
        f.write(line1)
    line2 = ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\"
    print(line2)
    with open(os.path.join(res_base_path,'res.txt'),'a') as f:
        f.write("\n"+line2)
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    if opts.uncertainty:
        res_path = opts.eval_out_dir
        uncertainty_map(res_path)
    else:
        evaluate(opts)
