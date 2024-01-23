# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class NYUDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        # self.K = np.array([[0.81, 0, 0.51, 0],
        #                    [0, 1.08, 0.53, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)

        self.K = np.array([[934.5331060961297 / 1280, 0, 646.162249922997 / 1280, 0],
                          [0, 934.6172960887559 / 960, 358.8661673371651 / 960, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]], dtype = np.float32)

        self.full_res_shape = (640, 480)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):

        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class NYURAWDataset(NYUDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(NYURAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, '.jpg')
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path
    
    def get_pose(self, folder, frame_index, side, do_flip):
        f_str = "{:04d}{}".format(frame_index, '.npy')
        pose_path = os.path.join(
            './splits/1204/cpose/', folder, f_str)
        pose = np.load(pose_path)
        #print(pose.shape)
        return pose
    
    '''
    def obtain_imu_data(self, file):
        gx, gy, gt = [], [], [] # z轴方向的移动相对于x,y轴小得多 所以忽略掉
        with open(file) as f:
            lines = f.read().splitlines()
            for i in range(2, len(lines)):
                row = lines[i].split(',')
                gx.append(float(row[0]))
                gy.append(-1 * float(row[1]))
                gt.append(int(row[-1]) / 1e6)
        gt = np.array(gt)
        gt = gt - gt[0] # 相对时间

        gxs, gys = [0] * (len(gx) + 1), [0]  * (len(gy) + 1) # gx, gy的前缀和
        for i in range(1, len(gx) + 1):
            gxs[i] = gxs[i - 1] + gx[i - 1]
            gys[i] = gys[i - 1] + gy[i - 1]
        
        index_list = [] # 相机是约1000 / 29.91毫秒拍摄一次，找到相机拍摄时间点在陀螺仪时间戳中的位置
        FRAME_UPPER_BOUND = 402
        target_index = 0

        # index_list索引为i的地方是i+1.png
        for timestamps in [1000 / 29.91 * i for i in range(1, FRAME_UPPER_BOUND)]:
            while(target_index < len(gt) and gt[target_index] < timestamps):
                target_index += 1
            index_list.append(target_index)
        return gxs, gys, gt, index_list

    def get_pose(self, folder, frame_index, side, do_flip, gxs, gys, index_list):
        #gxs, gys, gt, index_list = self.obtain_imu_data(file = f'../datasets/20231017/training/{folder}/imu.txt')

        pose = np.zeros((2, 4, 4))
        pose[0] = [[1,0,0,gxs[index_list[frame_index-1]] - gxs[index_list[frame_index]]],
                   [0,1,0,gys[index_list[frame_index-1]] - gys[index_list[frame_index]]],
                   [0,0,1,0],
                   [0,0,0,1]]
        pose[1] = [[1,0,0,gxs[index_list[frame_index+1]] - gxs[index_list[frame_index]]],
                   [0,1,0,gys[index_list[frame_index+1]] - gys[index_list[frame_index]]],
                   [0,0,1,0],
                   [0,0,0,1]]
        return pose
    '''

class NYUTestDataset(NYUDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(NYUTestDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:04d}{}".format(frame_index, '.png')
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

