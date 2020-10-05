from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import torch.nn.functional as F
import cv2
import PIL.Image as pil
import skimage.transform
import pdb

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class My_MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png',
                 refine=True,
                 crop_mode='c',
                 crop_h = None,
                 crop_w = None):
        super().__init__()

        self.refine = refine
        self.crop_mode = crop_mode
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # self.crop_h = [96,128,160,192,192]
        # self.crop_w = [192,256,384,448,640]
        self.crop_h = crop_h
        self.crop_w = crop_w

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = True

    
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        inputs["index"] = index

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        frame_index1 = float(line[1])
        frame_index2 = float(line[2])
        frame_index3 = float(line[3])
        frame_indexes = [frame_index1,frame_index2,frame_index3]
        frame_index = frame_index2

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for ind,i in enumerate(self.frame_idxs):
            
            inputs[("color", i, -1)] = self.get_color(folder, frame_indexes[ind], side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        
        if self.refine:
            for scale in range(5):
                K = self.K.copy()

                K[0, 0] *= self.width
                K[1, 1] *= self.height
                if self.crop_mode == 'b':
                    K[0, 2] *= self.crop_w[scale]
                    K[1, 2] = self.crop_h[scale] - 0.5*self.height
                else:
                    K[0, 2] *= self.crop_w[scale]
                    K[1, 2] *= self.crop_h[scale]

                inv_K = np.linalg.pinv(K)
                inputs[("K_r", scale)] = torch.from_numpy(K)
                inputs[("inv_K_r", scale)] = torch.from_numpy(inv_K)    
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            kernel = np.ones((3, 3), np.uint8)
            depth_gt_dilated = cv2.dilate(depth_gt, kernel)
            inputs["depth_gt_part"] = torch.from_numpy(depth_gt_dilated).unsqueeze(0)
            #inputs["mask"] = torch.from_numpy(mask).unsqueeze(0)
            #inputs["mask_edge"] = torch.from_numpy(mask_edge).unsqueeze(0)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))      
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError



class MyDataset(My_MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.K = np.array([[1.624, 0, 0.5, 0],
                           [0, 2.433, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1536, 1024)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:.6f}.png".format(frame_index)
        image_path = os.path.join(
            self.data_path,
            folder,
            'rgb',
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:.6f}.npy".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            'depth_sparse',
            f_str)
        depth_gt = np.load(depth_path).astype(np.float32)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        depth_gt[depth_gt>600] = 0
        depth_gt[depth_gt<10] = 0
        return depth_gt



        

