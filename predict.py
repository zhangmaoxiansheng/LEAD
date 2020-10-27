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
from tqdm import tqdm
import cv2
import PIL.Image as pil
from torchvision import transforms
import pdb

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



def load_model_dict(model_path,model):
    pretrained_model_dict = torch.load(model_path,map_location=torch.device("cpu"))
    model_dict = model.state_dict()
    model.load_state_dict({k.replace('module.',''): v for k, v in pretrained_model_dict.items() if k.replace('module.','') in model_dict})


def predict(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"


    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    refine = opt.refine or opt.dropout

    if refine:
        opt.refine_stage = list(range(opt.refine_stage))
        crop_h = [128,192,256]
        crop_w = [192,288,384]
    else:
        crop_h = None
        crop_w = None
    encoder_dict = torch.load(encoder_path,map_location=torch.device("cpu"))

    image_path = opt.rgb_path
    depth_path = opt.dep_path
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size

    feed_height = opt.height
    feed_width = opt.width
    input_image_resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)

    input_color = transforms.ToTensor()(input_image_resized).unsqueeze(0).cuda()

    depth_input = np.load(depth_path).astype(np.float32)
    depth_input_origin = depth_input.copy()
    depth_input[depth_input>800] = 0
    depth_input[depth_input<2.5] = 0
    depth_input = np.ascontiguousarray(depth_input)
    depth_input = torch.from_numpy(depth_input).unsqueeze(0).cuda()
    depth_input = depth_input.unsqueeze(0)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc,refine=refine)

    model_dict = encoder.state_dict()
    
    encoder.load_state_dict({k.replace('module.',''): v for k, v in encoder_dict.items() if k.replace('module.','') in model_dict})
    depth_decoder_dict = torch.load(decoder_path,map_location=torch.device("cpu"))
    depth_decoder.load_state_dict({k.replace('module.',''): v for k, v in depth_decoder_dict.items()})

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()
    if refine:
        renet_path = os.path.join(opt.load_weights_folder, "mid_refine.pth")
        if opt.crop_mode == 'b':
            mid_refine = networks.Iterative_Propagate_old(crop_h,crop_w,opt.crop_mode)
        else:
            if opt.refine_model == '3D':
                mid_refine = networks.Iterative_3DPropagate(crop_h,crop_w,opt.crop_mode)
            elif opt.refine_model == '18':
                mid_refine = networks.Iterative_Propagate2(crop_h,crop_w,opt.crop_mode,False)
            else:
                mid_refine = networks.Iterative_Propagate(crop_h,crop_w,opt.crop_mode)
        
        load_model_dict(renet_path,mid_refine)
        mid_refine.cuda()
        mid_refine.eval()

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
        
        depth_part_gt =  F.interpolate(depth_input, [opt.height, opt.width], mode="nearest")
        input_rgbd = torch.cat((input_color,depth_part_gt),1)
        features_init = encoder(input_rgbd)
        output = depth_decoder(features_init)
        #output = depth_decoder(encoder(input_color))
        output_disp = output[("disp", 0)]

        if refine and not opt.dropout:
            disp_blur = output[("disp", 0)]
            features = output["disp_feature"]
            disp_part_gt = depth_to_disp(depth_part_gt ,opt.min_depth,opt.max_depth)
            output = mid_refine(features,disp_blur, disp_part_gt,input_color,opt.refine_stage)
            final_stage = opt.refine_stage[-1]
            output_disp = output[("disp", final_stage)]
        if opt.dropout and not opt.eval_step:
            #baseway
            disp_blur = output[("disp", 0)]
            outputs2 = depth_ref(features_init)
            disp_part_gt = depth_to_disp(depth_part_gt ,opt.min_depth,opt.max_depth)
            output = mid_refine(outputs2["disp_feature"],disp_blur, disp_part_gt,input_color,opt.refine_stage)
            final_stage = opt.refine_stage[-1]
            output_disp = output[("disp", final_stage)]
        if opt.eval_step and opt.dropout:
            outputs2 = {}
            output_f = {}

            disp_blur = output[("disp", 0)]
            
            disp_part_gt = depth_to_disp(depth_part_gt ,opt.min_depth,opt.max_depth)
            iter_time=opt.iter_time
            out = []
            
            for i in opt.refine_stage:
                if i == 0:
                    dep_last = disp_part_gt
                else:
                    dep_last = outputs2[("disp",i-1)]
                for it in range(iter_time):
                    output_f = depth_decoder(features_init,True)
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
            
        pred_disp, depth = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)
        depth = depth.cpu()[:, 0].numpy()

    depth = depth.squeeze()
    depth = cv2.resize(depth,(1536,1024))
    pdb.set_trace()
    

    gt = np.load('fuck_gt.npy').squeeze()
    
    mask2 = depth_input_origin > 0
    gt[mask2] = depth_input_origin[mask2]
    depth[mask2] = depth_input_origin[mask2]
    
    mask = gt>0
    gt = gt[mask]
    depth = depth[mask]
    print(compute_errors(gt,depth))

    pdb.set_trace()

    plt.imsave('result.png',depth,cmap='plasma')
    
        

if __name__ == "__main__":
    options = MonodepthOptions()
    predict(options.parse())
