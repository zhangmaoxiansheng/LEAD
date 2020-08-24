from layers import disp_to_depth, depth_to_disp
import numpy as np
import os
import cv2
base_path = './result_test'
def l1_loss(img1,gt):
    mask = gt>0
    return np.mean(np.abs(img1[mask] - gt[mask]))
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
def all_error(depth,gt):
    errors = []
    for i in range(gt.shape[0]):
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        gt_depth = gt[i]
        gt_depth = cv2.resize(gt_depth, (1242, 375))
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = depth[i]
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
    
    line1 = "\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
    print(line1)

    line2 = ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\"
    print(line2)

    print("\n-> Done!")
    return mean_errors

gt = np.load(os.path.join(base_path,'gt.npy'))
output_eval = 1 / np.load(os.path.join(base_path,'output_disp.npy'))

output_offline = np.load(os.path.join(base_path,'offline_depres.npy'))
print(l1_loss(output_eval,output_offline))

all_error(output_eval,gt)
all_error(output_offline,gt)

error_eval = np.load(os.path.join(base_path,'error_save.npy'))
error_offline = np.load(os.path.join(base_path,'error_offline.npy'))
print(np.mean(np.abs(error_eval-error_offline)))
print(error_eval[0])
print(error_offline[0])
