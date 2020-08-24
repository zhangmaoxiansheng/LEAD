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


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        #filenames = readlines(os.path.join(splits_dir, opt.eval_split, "val_files_p.txt"))
        filenames = readlines(os.path.join(splits_dir, "eigen_benchmark", "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        refine = opt.refine or opt.dropout

        if refine:
            opt.refine_stage = list(range(opt.refine_stage))
            crop_h = [96,128,160,192]
            crop_w = [192,256,384,640]
            if len(opt.refine_stage) > 4:
                if opt.crop_mode == 'b':
                    crop_h = [128,168,192,192,192]
                    crop_w = [192,256,384,448,640]
                else:
                    crop_h = [96,128,160,192,192]
                    crop_w = [192,256,384,448,640]
        else:
            crop_h = None
            crop_w = None
        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIDepthDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, refine=opt.refine, crop_mode=opt.crop_mode, crop_h=crop_h, crop_w=crop_w)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc,refine=refine)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()
        if refine and not opt.dropout:
            renet_path = os.path.join(opt.load_weights_folder, "mid_refine.pth")
            if opt.refine_model == 'i':
                mid_refine = networks.Iterative_Propagate(crop_h,crop_w,opt.crop_mode)
            elif opt.refine_model == 'io':
                mid_refine = networks.Iterative_Propagate_old(crop_h,crop_w,opt.crop_mode)
            else:
                mid_refine = networks.Simple_Propagate(crop_h,crop_w,opt.crop_mode)
            mid_refine.load_state_dict(torch.load(renet_path))
            mid_refine.cuda()
            mid_refine.eval()
        if opt.dropout:
            depth_ref_path = os.path.join(opt.load_weights_folder, "depth_ref.pth")
            renet_path = os.path.join(opt.load_weights_folder, "mid_refine.pth")
            
            if opt.refine_model == 'i':
                mid_refine = networks.Iterative_Propagate(crop_h,crop_w,opt.crop_mode)
            elif opt.refine_model == 'io':
                mid_refine = networks.Iterative_Propagate_old(crop_h,crop_w,opt.crop_mode)
            else:
                mid_refine = networks.Simple_Propagate(crop_h,crop_w,opt.crop_mode)
            depth_ref = networks.DepthDecoder(encoder.num_ch_enc,refine=refine)
            
            mid_refine.load_state_dict(torch.load(renet_path))
            mid_refine.cuda()
            mid_refine.eval()
            depth_ref.load_state_dict(torch.load(depth_ref_path))
            depth_ref.cuda()
            depth_ref.eval()

        pred_disps = []
        gt = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        batch_index = 0
        if refine:
            output_save = {}
            for i in opt.refine_stage:
                output_save[i] = []
            output_part_gt = []
            error_saved = []
        with torch.no_grad():
            for data in dataloader:
                batch_index += 1
                gt.append(data["depth_gt"].cpu()[:,0].numpy())
                input_color = data[("color", 0, 0)].cuda()
                for key, ipt in data.items():
                    data[key] = ipt.cuda()
                depth_part_gt =  F.interpolate(data["depth_gt_part"], [opt.height, opt.width], mode="nearest")
                input_rgbd = torch.cat((input_color,depth_part_gt),1)
                features_init = encoder(input_rgbd)
                output = depth_decoder(features_init)
                #output = depth_decoder(encoder(input_color))
                output_disp = output[("disp", 0)]

                if refine and not opt.dropout:
                    disp_blur = output[("disp", 0)]
                    features = output["disp_feature"]
                    depth_part_gt = F.interpolate(data["depth_gt_part"], [opt.height, opt.width], mode="nearest")
                    disp_part_gt = depth_to_disp(depth_part_gt ,opt.min_depth,opt.max_depth)
                    output = mid_refine(features,disp_blur, disp_part_gt,input_color,opt.refine_stage)
                    final_stage = opt.refine_stage[-1]
                    output_disp = output[("disp", final_stage)]
                if opt.dropout and not opt.eval_step:
                    #baseway
                    disp_blur = output[("disp", 0)]
                    outputs2 = depth_ref(features_init)
                    depth_part_gt = F.interpolate(data["depth_gt_part"], [opt.height, opt.width], mode="nearest")
                    disp_part_gt = depth_to_disp(depth_part_gt ,opt.min_depth,opt.max_depth)
                    output = mid_refine(outputs2["disp_feature"],disp_blur, disp_part_gt,input_color,opt.refine_stage)
                    final_stage = opt.refine_stage[-1]
                    output_disp = output[("disp", final_stage)]
                if opt.eval_step and opt.dropout:
                    outputs2 = {}
                    output_f = {}

                    disp_blur = output[("disp", 0)]
                    depth_part_gt = F.interpolate(data["depth_gt_part"], [opt.height, opt.width], mode="nearest")
                    
                    disp_part_gt = depth_to_disp(depth_part_gt ,opt.min_depth,opt.max_depth)
                    iter_time=opt.iter_time
                    out = []
                    
                    for i in opt.refine_stage:
                        if i == 0:
                            dep_last = disp_part_gt
                        else:
                            dep_last = outputs2[("disp",i-1)]
                        for it in range(iter_time):
                            output_f = depth_ref(features_init,True)
                            stage_output,error = mid_refine.eval_step(output_f["disp_feature"],disp_blur,disp_part_gt,input_color,i,dep_last,depth_part_gt)
                            output_save[i].append(stage_output.cpu()[:, 0].numpy())
                            error_saved.append(error.cpu().numpy())
                            if it == 0:
                                outputs2[("disp", i)] = stage_output
                                best_error = error
                            elif error <  best_error:
                                outputs2[("disp", i)] = stage_output
                                best_error = error
                    
                    final_stage = opt.refine_stage[-1]
                    output_disp = outputs2[("disp", final_stage)]
                    output_part_gt.append(depth_part_gt.cpu()[:, 0].numpy())
                    
                pred_disp, _ = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
        pred_disps = np.concatenate(pred_disps,axis=0)
        gt = np.concatenate(gt,axis=0)
        if opt.save_pred_disps:
            output_part_gt = np.concatenate(output_part_gt,axis=0)
            # for i in opt.refine_stage:
            #     output_save[i] = np.concatenate(output_save[i],axis=0)
    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:

        # output_path = os.path.join(
        #     opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        # print("-> Saving predicted disparities to ", output_path)
        # np.save(output_path, pred_disps)
        save_base_path = os.path.join(opt.load_weights_folder,'./result')
        if not os.path.exists(save_base_path):
            os.mkdir(save_base_path)

        #save error
        np.save(os.path.join(save_base_path,'error_save.npy'),error_saved)
        #save gt
        np.save(os.path.join(save_base_path,'gt.npy'),gt)
        #save part gt
        np.save(os.path.join(save_base_path,'part_gt.npy'),output_part_gt)
        #save output
        np.save(os.path.join(save_base_path,'output_disp.npy'),pred_disps)
        for i in opt.refine_stage:
            save_list = output_save[i]
            
            for count,ind in enumerate(range(0,len(save_list),iter_time)):
                save_image_set = np.concatenate(save_list[ind:ind+iter_time],axis=0)
                np.save(os.path.join(save_base_path,'%d_stage%d.npy'%(count,i)),save_image_set)

    if opt.uncertainty:
        #save the uncertainty map
        save_base_path = os.path.join(opt.load_weights_folder,'./result_uncert')
        save_list = output_save[opt.refine_stage[-1]]
        for count,ind in enumerate(range(0,len(save_list),iter_time)):
            image_set = np.concatenate(save_list[ind:ind+iter_time],axis=0)
            uncert = np.std(image_set,0)
            plt.imsave(os.path.join(save_base_path,'{}_uncert.png'.format(i)),uncert,cmap='Greys')
    if opt.save_cmap:
        #save the cmap of the depth
        #and the outputs
        save_base_path = os.path.join(opt.load_weights_folder,'./result_cmap')
        np.save(os.path.join(save_base_path,'output_disp.npy'),pred_disps)
        for i in range(pred_disps.shape[0]):
            pred_depth = 1 / pred_disps[i]
            plt.imsave(os.path.join(save_base_path,'{}.png'.format(i)),pred_depth,cmap='plasma')

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()
    
    gt_depths = gt
    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_depth = cv2.resize(gt_depth, (1242, 375))
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp
        
        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth_o = crop_center(pred_depth)
        pred_depth = pred_depth[mask]
        gt_depth_part = crop_center(gt_depth)
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if opt.median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio
        if opt.center_median_scaling:
            mask2 = gt_depth_part>0
            gt_depth_part = gt_depth_part[mask2]
            ratio = np.median(gt_depth_part) / np.median(pred_depth_o[mask2])
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if opt.median_scaling or opt.center_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    line1 = "\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
    print(line1)
    with open(os.path.join(opt.load_weights_folder,'res.txt'),'a') as f:
        f.write(line1)
    line2 = ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\"
    print(line2)
    with open(os.path.join(opt.load_weights_folder,'res.txt'),'a') as f:
        f.write("\n"+line2)
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
