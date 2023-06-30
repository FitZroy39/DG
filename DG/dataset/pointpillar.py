import numpy as np
import math
import torch
from glob import glob
from torch import nn
import torch.utils.data
import random
import os

from random import shuffle
from configs import SELF_CAR_EXTENTS,CATEGORY_TO_ID
import open3d as o3d
from numba import njit
from core.voxel_generator import VoxelGenerator
from configs import SELF_CAR_EXTENTS, AREA_EXTENTS, voxel_size, IMG_SIZE
from dataset.preprocess import random_flip_along_x, global_rotation, global_scaling, global_translate, jitter_point
    #random_flip, global_rotation_v2, global_scaling_v2, global_translate

def load_pcd_to_ndarray(pcd_path):
    with open(pcd_path) as f:
        while True:
            ln = f.readline().strip()
            if ln.startswith('DATA'):
                break

        points = np.loadtxt(f, dtype=np.float32)
        points = points[:, 0:4]
        return points

def read_pcd(file_path):
    if file_path.endswith('.pcd'):
        points = load_pcd_to_ndarray(file_path)
        #pcd = o3d.io.read_point_cloud(file_path)
        #points = np.asarray(pcd.points)
    elif file_path.endswith('.bin'):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points



#SELF_CAR_EXTENTS = [[-7.18, 0],[-1.85, 1.85]]
#AREA_EXTENTS = [[-50, 100], [-50, 50], [-1, 2]]#[[-40, 97.6], [-49.6,49.6], [-0.35, 4.2]]
#voxel_size = (0.5, 0.3, 3)#0.1
#log_norm = 4
#num_slice = 5
point_cloud_range = [AREA_EXTENTS[0][0], AREA_EXTENTS[1][0], AREA_EXTENTS[2][0],
                     AREA_EXTENTS[0][1], AREA_EXTENTS[1][1], AREA_EXTENTS[2][1]]#[-50, -50, -1, 100, 50, 2]
max_num_points = 100
#bev_x_h = round((AREA_EXTENTS[0][1] - AREA_EXTENTS[0][0]) / voxel_size[0])
#bev_y_w = round((AREA_EXTENTS[1][1] - AREA_EXTENTS[1][0]) / voxel_size[1])
#grid_size = (bev_x_h, bev_y_w)

class POINTPILLAR_Dataset(torch.utils.data.Dataset):
    def __init__(self, img_lists, status, transforms=None, config=None):
        self.transforms = transforms
        dev = config['dev_mode']
        self.config = config
        self.status = status

        if status == 'Train':
            self.imgs = img_lists

            # shuffle
            shuffle(self.imgs)
        else:
            self.imgs = img_lists


    def __getitem__(self, idx):
        # load images ad masks
        #img, target = self.get_ele(idx)
        data = self.get_ele(idx)

        #if self.transforms is not None:
        #    img, target = self.transforms(img, target)

        return data#img, target

    ''''
    def data_augment(self, points, gt_gridmap):
        gt_gridmap, points = random_flip(gt_gridmap, points)
        gt_gridmap, points = global_rotation_v2(gt_gridmap, points, min_rad=-np.pi / 4,
                       max_rad=np.pi / 4)
        gt_gridmap, points = global_scaling_v2(gt_gridmap, points, min_scale=0.95, max_scale=1.05)

        # Global translation
        gt_gridmap, points = global_translate(gt_gridmap, points, noise_translate_std=(0.2, 0.2, 0.2))
        return points, gt_gridmap
    '''

    def data_augment(self, points, gt_gridmap):
        #gt_gridmap, points, _ = random_flip_along_x(gt_gridmap, points, None)
        #gt_gridmap, points, _ = global_rotation(gt_gridmap, points, None, rot_range=[-0.78539816, 0.78539816])
        #gt_gridmap, points, _ = global_scaling(gt_gridmap, points, None, scale_range=[0.95, 1.05])
        #gt_gridmap, points, _ = global_translate(gt_gridmap, points, None, noise_translate_std=(0.2, 0.2, 0.2))
        #points = jitter_point(points)
        return points, gt_gridmap
    def order_img_list(self):
        img_list = []
        for key, ele in self.imgs.items():
            img_list.extend(ele)
        shuffle(img_list)
        self.imgs = img_list
        print('lenth of img_list:', len(self.imgs))

    def get_ele(self, idx):
        img_path = self.imgs[idx]
        #points = np.loadtxt(img_path)
        if img_path.endswith('.txt'):
            points = np.loadtxt(img_path)
        else:
            points = read_pcd(img_path)



        #target = {}
        if self.config['multiple_classes']:
            label_path = img_path.replace('images_fine', 'annotations0503_fine')
        else:
            label_path = img_path.replace('images', 'annotations')
        if label_path.endswith('.pcd'):
            label_path = label_path.replace(".pcd", ".txt")
        elif label_path.endswith('.bin'):
            label_path = label_path.replace(".bin", ".txt")
        #gt_info = np.loadtxt(label_path, dtype=int)
        gt_info = np.loadtxt(label_path, dtype=str)
        bev_x_h = IMG_SIZE[0]#round((AREA_EXTENTS[0][1] - AREA_EXTENTS[0][0]) / voxel_size[0])
        bev_y_w = IMG_SIZE[1]#round((AREA_EXTENTS[1][1] - AREA_EXTENTS[1][0]) / voxel_size[1])
        gt_semantic_seg = np.zeros((bev_x_h, bev_y_w, 1), dtype=np.int32)#np.zeros((bev_x_h, bev_y_w, 1), dtype=np.int32)
        length = len(gt_info)

        for i in range(length):
            x = int(gt_info[i][0])
            y = int(gt_info[i][1])
            if self.config['multiple_classes']:
                cate = gt_info[i][2].lower()
                #if cate not in CATEGORY_TO_ID.keys():
                    #print('************',cate,label_path)
                assert cate in CATEGORY_TO_ID.keys()
                #if cate in CATEGORY_TO_ID.keys():
                cate_id = CATEGORY_TO_ID[cate]
                gt_semantic_seg[y, x, 0] = cate_id
                #gt_semantic_seg[x, y, 0] = cate_id
            else:
                is_obstacle = int(gt_info[i][2])
                is_valid = int(gt_info[i][4])
                if is_obstacle == 1 and is_valid == 0:
                    #gt_semantic_seg[x, y, 0] = 1
                    gt_semantic_seg[y, x, 0] = 1
                    # gt_semantic_seg[x, y, 1] = 255
                    # gt_semantic_seg[x, y, 2] = 255


        max_voxels = 12000
        if self.status == 'Train':
            points, gt_semantic_seg = self.data_augment(points, gt_semantic_seg)
            max_voxels = 40000

        target = gt_semantic_seg.squeeze().astype(np.uint8)

        voxel_gen = VoxelGenerator(voxel_size,
                                   point_cloud_range,
                                   max_num_points,
                                   max_voxels=max_voxels,#12000,
                                   self_car_extend=SELF_CAR_EXTENTS)
        voxels, coors, num_points_per_voxel = voxel_gen.generate(points, max_voxels=max_voxels)#12000)
        #print(voxels.shape, coors.shape, num_points_per_voxel.shape, target.shape)
        data = {}
        data['voxels'] = voxels
        data['coordinates'] = coors
        data['num_points'] = num_points_per_voxel
        data['targets'] = target
        return data#, target

    def __len__(self):
        return len(self.imgs)