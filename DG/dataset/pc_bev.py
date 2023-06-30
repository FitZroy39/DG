import numpy as np
import math
import torch
import torch.utils.data
import os

from random import shuffle
import open3d as o3d
from numba import njit

import sys
sys.path.append("..") 
from configs import SELF_CAR_EXTENTS,AREA_EXTENTS,voxel_size,log_norm,num_slice, CATEGORY_TO_ID, IMG_SIZE, DETECTION_AREA_EXTENTS

# only used when observability channel is added to bev
@njit(cache=True, fastmath=True)
def cal_grid_pcd(points):
    grid_pcd_cnt = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.int32)
    length = len(points)
    for i in range(length):
        x = points[i][0]
        y = points[i][1]
        z = points[i][2]
        if AREA_EXTENTS[0][0] < x < AREA_EXTENTS[0][1]:
            if AREA_EXTENTS[1][0] < y < AREA_EXTENTS[1][1]:
                if 0.2< z < 2:
                    gx = int((x - AREA_EXTENTS[0][0]) / voxel_size[0])
                    gy = int((y - AREA_EXTENTS[1][0]) / voxel_size[1])
                    grid_pcd_cnt[gx][gy] += 1
    return grid_pcd_cnt
# only used when observability channel is added to bev
@njit(cache=True, fastmath=True)
def is_line_intersect_gird(sx,sy,ex,ey,x1,y1,x2,y2):
    k = (ey - sy) / (ex - sx)
    b = sy - k*sx
    #判断与上边是否相交
    x_intersct = (y1 - b) / k
    if x1 <= x_intersct <= x2:
        return True
    # 判断与下边是否相交
    x_intersct = (y2 - b) / k
    if x1 <= x_intersct <= x2:
        return True
    # 判断与左边是否相交
    y_intersct = k * x1 + b
    if y2 <= y_intersct <= y1:
        return True
    # 判断与右边是否相交
    y_intersct = k * x2 + b
    if y2 <= y_intersct <= y1:
        return True
    return False
# only used when observability channel is added to bev
@njit(cache=True, fastmath=True)
def cal_observabilty(points, grid_pcd_cnt):
    obs = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.float32)
    for p in points:
        px = p[0]
        py = p[1]
        pz = p[2]
        if px < AREA_EXTENTS[0][0] or px > AREA_EXTENTS[0][1] or py < AREA_EXTENTS[1][0] or py > AREA_EXTENTS[1][1] or \
            pz < 0.2 or pz > AREA_EXTENTS[2][1] or \
                (SELF_CAR_EXTENTS[0][0]< px < SELF_CAR_EXTENTS[0][1] and SELF_CAR_EXTENTS[1][0]< py < SELF_CAR_EXTENTS[1][1]):
            continue
        first_visit_pcd_flag = 0
        if px > 0 and py > 0:
            for i in np.arange(0, px, voxel_size[0]):
                for j in np.arange(0.1, py, voxel_size[1]):
                    x1 = i
                    y1 = j + voxel_size[1]
                    x2 = i + voxel_size[0]
                    y2 = j
                    is_occluded = is_line_intersect_gird(0, 0, px, py, x1, y1, x2, y2)
                    if is_occluded:
                        idx = int((i - AREA_EXTENTS[0][0]) / voxel_size[0])
                        idy = int((j - AREA_EXTENTS[1][0]) / voxel_size[1])
                        if first_visit_pcd_flag == 0:
                            obs[idx, idy, 0] += 1
                            if grid_pcd_cnt[idx, idy] > 0:
                                first_visit_pcd_flag = 1
                                break
                if first_visit_pcd_flag == 1:
                    break
        elif px < 0 and py > 0:
            for i in np.arange(0, px, -voxel_size[0]):
                for j in np.arange(0.1, py, voxel_size[1]):
                    x1 = i - voxel_size[0]
                    y1 = j + voxel_size[1]
                    x2 = i
                    y2 = j
                    is_occluded = is_line_intersect_gird(0, 0, px, py, x1, y1, x2, y2)
                    if is_occluded:
                        idx = int((i - AREA_EXTENTS[0][0]) / voxel_size[0])
                        idy = int((j - AREA_EXTENTS[1][0]) / voxel_size[1])
                        if first_visit_pcd_flag == 0:
                            obs[idx, idy, 0] += 1
                            if grid_pcd_cnt[idx, idy] > 0:
                                first_visit_pcd_flag = 1
                                break
                if first_visit_pcd_flag == 1:
                    break
        elif px < 0 and py < 0:
            for i in np.arange(0, px, -voxel_size[0]):
                for j in np.arange(-0.2, py, -voxel_size[1]):
                    x1 = i - voxel_size[0]
                    y1 = j
                    x2 = i
                    y2 = j - voxel_size[1]
                    is_occluded = is_line_intersect_gird(0, 0, px, py, x1, y1, x2, y2)
                    if is_occluded:
                        idx = int((i - AREA_EXTENTS[0][0]) / voxel_size[0])
                        idy = int((j - AREA_EXTENTS[1][0]) / voxel_size[1])
                        if first_visit_pcd_flag == 0:
                            obs[idx, idy, 0] += 1
                            if grid_pcd_cnt[idx, idy] > 0:
                                first_visit_pcd_flag = 1
                                break
                if first_visit_pcd_flag == 1:
                    break
        elif px > 0 and py < 0:
            for i in np.arange(0, px, voxel_size[0]):
                for j in np.arange(-0.2, py, -voxel_size[1]):
                    x1 = i
                    y1 = j
                    x2 = i + voxel_size[0]
                    y2 = j - voxel_size[1]
                    is_occluded = is_line_intersect_gird(0, 0, px, py, x1, y1, x2, y2)
                    if is_occluded:
                        idx = int((i - AREA_EXTENTS[0][0]) / voxel_size[0])
                        idy = int((j - AREA_EXTENTS[1][0]) / voxel_size[1])
                        if first_visit_pcd_flag == 0:
                            obs[idx, idy, 0] += 1
                            if grid_pcd_cnt[idx, idy] > 0:
                                first_visit_pcd_flag = 1
                                break
                if first_visit_pcd_flag == 1:
                    break

    return obs




def read_pcd(file_path):
    if file_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    elif file_path.endswith('.bin'):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

@njit(cache=True, fastmath=True)
def extract_bev_map_kernel(
    pc, bev_map, point_extents, voxel_size, height_per_slice, num_slices, log_norm
):
    point_num = pc.shape[0]
    for i in range(point_num):
        x = pc[i, 0]
        y = pc[i, 1]
        z = pc[i, 2]
        # 在检测范围内area_extent
        if (
            x > point_extents[0, 0]
            and x < point_extents[0, 3]
            and y > point_extents[1, 0]
            and y < point_extents[1, 3]
            and z > point_extents[2, 0]
            and z < point_extents[2, 1]
        ):
            # 在自车范围外
            if not (
                x > point_extents[0, 1]
                and x < point_extents[0, 2]
                and y > point_extents[1, 1]
                and y < point_extents[1, 2]
            ):
                # self.point_extents = [[-25, -7.18, 0, 60], [-40.5, -1.85, 1.85, 40.5],[-0.3, 2.2]]
                # x - point_extents[0, 0] 表达的是在 检测范围下 的栅格坐标 x_coord
                x_coord = int((x - point_extents[0, 0]) / voxel_size[0])
                y_coord = int((y - point_extents[1, 0]) / voxel_size[1])
                # (height_high - height_low) / num_slices 2.2- -0.3 / 5 (0.5)
                height_ratio = (z - point_extents[2, 0]) / height_per_slice
                height_coord = int(height_ratio)
                # 记录与相应高度层的最大差值
                bev_map[height_coord, x_coord, y_coord] = max(
                    bev_map[height_coord, x_coord, y_coord], height_ratio - height_coord
                )
                # bev_map[num_slices, x_coord, y_coord]表示密度层
                bev_map[num_slices, x_coord, y_coord] += 1

    x_range = bev_map.shape[1]
    y_range = bev_map.shape[2]
    for x_coord in range(x_range):
        for y_coord in range(y_range):
            points_num = bev_map[num_slices, x_coord, y_coord]
            if points_num > 0:
                bev_map[num_slices, x_coord, y_coord] = min(
                    1, math.log(points_num + 1) / log_norm
                )
            # 获得点云坐标图
            bev_map[num_slices+1, x_coord, y_coord] = x_coord / x_range
            bev_map[num_slices+2, x_coord, y_coord] = y_coord / y_range
    return bev_map.transpose((1, 2, 0))



class ExtractBevFeature(object):
    """Extract Bird eye view feature for point cloud"""

    def __init__(
     self,
     area_extents,
     self_car_extents,
     num_slices,
     voxel_size,
     #anchor_strides,
     log_norm,
     fp_16=False,
     remain_points=False):
        # 这里的*2是将维度扩大两倍
        # x_range[0]，x_range[3]是x轴上检测范围上下限
        # x_range[1], x_range[2]是x轴自车在x轴检测范围的上下限
        x_range, y_range = area_extents[0] * 2, area_extents[1] * 2 
        x_range[1:3], y_range[1:3] = self_car_extents[0], self_car_extents[1]
        z_range = area_extents[2] * 2

        point_extents = [x_range, y_range, z_range]
        self.point_extents = np.array(point_extents)
        height_low, height_high = area_extents[2]
        bev_z = int(num_slices + 3)#int(num_slices + 1)
        self.area_extents = area_extents
        self.num_slices = num_slices
        self.bev_shape = (bev_z, IMG_SIZE[0], IMG_SIZE[1])
        self.voxel_size = voxel_size
        self.height_per_slice = (height_high - height_low) / num_slices
        self.log_norm = log_norm
        self.fp_16 = fp_16
        self.remain_points = remain_points

    def __call__(self, points):
        bev_map = np.zeros(self.bev_shape, dtype=np.float32)
        bev_map = extract_bev_map_kernel(
            points,
            bev_map,
            self.point_extents,
            self.voxel_size,
            self.height_per_slice,
            self.num_slices,
            self.log_norm,
        )
        if self.fp_16:
            bev_map = bev_map.astype(np.float16)
        return bev_map

# only used for multiclass, and is optional
def upsample_few_category(points, gt_seg, fid, category, repeat_times, labeled_bbox_root):
    assert category in ['car', 'pedestrian', 'block']
    labeled_bbox_name = os.path.join(labeled_bbox_root, fid)
    labeled_bbox = np.loadtxt(labeled_bbox_name, dtype=str)
    if len(labeled_bbox.shape) == 1:
        labeled_bbox = labeled_bbox.reshape(-1, 5)
        #return points, gt_seg, False

    cate_bbox_mask = (labeled_bbox[:,4] == category)

    cate_bbox = labeled_bbox[cate_bbox_mask]
    cate_cnt = len(cate_bbox)
    if cate_cnt == 0:
        return points, gt_seg, False
    for i in range(repeat_times):
        select_id = np.random.randint(0, cate_cnt)
        minx, maxx, miny, maxy, _ = cate_bbox[select_id]
        minx, maxx, miny, maxy = int(minx), int(maxx)+1, int(miny), int(maxy)+1
        if maxx - minx > 80 or maxy - miny > 80:
            continue
        new_box_sum = -1
        loop_cnt = 0
        while new_box_sum != 0 and loop_cnt < 200:
            if minx > 100 and miny > 150:
                sx = np.random.randint(110, 220)
                sy = np.random.randint(160, 220)
            elif maxx < 100 and miny > 150:
                sx = np.random.randint(10, 80)
                sy = np.random.randint(160, 220)
            elif maxx < 100 and maxy < 150:
                sx = np.random.randint(10, 80)
                sy = np.random.randint(10, 140)
            elif minx > 100 and maxy < 150:
                sx = np.random.randint(110, 220)
                sy = np.random.randint(10, 140)
            else:
                sx = minx
                sy = miny

            new_box = gt_seg[sx:sx+maxx-minx, sy:sy+maxy-miny]
            new_box_sum = np.sum(new_box)
            loop_cnt += 1
        if loop_cnt >= 100:
            return points, gt_seg, False
        gt_seg[sx:sx+maxx-minx, sy:sy+maxy-miny] = gt_seg[minx:maxx, miny:maxy]

        range_minx = minx * voxel_size[0] + AREA_EXTENTS[0][0]
        range_maxx = maxx * voxel_size[0] + AREA_EXTENTS[0][0]
        range_miny = miny * voxel_size[1] + AREA_EXTENTS[1][0]
        range_maxy = maxy * voxel_size[1] + AREA_EXTENTS[1][0]
        range_sx = sx * voxel_size[0] + AREA_EXTENTS[0][0]
        range_sy = sy * voxel_size[1] + AREA_EXTENTS[1][0]

        repeat_points_mask1 = (range_minx<=points[:,0]) & (points[:,0]<=range_maxx)
        repeat_points_mask2 = (range_miny<=points[:,1]) & (points[:,1]<=range_maxy)
        repeat_points_mask = repeat_points_mask1 & repeat_points_mask2
        repeat_points = points[repeat_points_mask]
        repeat_points[:,0] += (range_sx-range_minx)
        repeat_points[:,1] += (range_sy-range_miny)
        points = np.concatenate([points, repeat_points], axis=0)

    return points, gt_seg, True

# only used for multiclass, and is optional
def generate_target_map(label_path, multiple_classes=True):
    gt_info = np.loadtxt(label_path, dtype=str)
    gt_semantic_seg = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.int32)
    length = len(gt_info)
    for i in range(length):
        x = int(gt_info[i][0])
        y = int(gt_info[i][1])
        if multiple_classes:
            cate = gt_info[i][2].lower()
            assert cate in CATEGORY_TO_ID.keys()
            cate_id = CATEGORY_TO_ID[cate]
            gt_semantic_seg[x, y, 0] = cate_id
        else:
            is_obstacle = int(gt_info[i][2])
            is_valid = int(gt_info[i][4])
            if is_obstacle == 1 and is_valid == 0:
                gt_semantic_seg[x, y, 0] = 1

    target = gt_semantic_seg.squeeze().astype(np.uint8)
    return target
# only used for multiclass, and is optional
def upsample_few_category_pro(points, gt_seg, category, pcd_dir, label_dir, repeat_times=10, max_loop=100):
    assert category in ['car', 'pedestrian', 'block']
    labeled_bbox_root = "/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev_10_classes0101/labeled_bbox_0503_{}_extra".format(category)
    fid_list = os.listdir(labeled_bbox_root)

    for i in range(repeat_times):
        fid = np.random.choice(fid_list)
        labeled_bbox_name = os.path.join(labeled_bbox_root, fid)
        labeled_bbox = np.loadtxt(labeled_bbox_name, dtype=str)
        if len(labeled_bbox.shape) == 1:
            labeled_bbox = labeled_bbox.reshape(-1, 5)
        cate_bbox = labeled_bbox#[cate_bbox_mask]
        cate_cnt = len(cate_bbox)

        extra_pcd_path = os.path.join(pcd_dir, fid.replace('.txt', '.bin'))
        extra_label_path = os.path.join(label_dir, fid)
        if not os.path.exists(extra_pcd_path):
            print('{} not exist'.format(extra_pcd_path))
            continue
            #return points, gt_seg, False
        if not os.path.exists(extra_label_path):
            print('{} not exist'.format(extra_label_path))
            continue
        extra_points = read_pcd(extra_pcd_path)
        extra_gt_seg = generate_target_map(extra_label_path, multiple_classes=True)

        select_id = np.random.randint(0, cate_cnt)
        minx, maxx, miny, maxy, _ = cate_bbox[select_id]
        minx, maxx, miny, maxy = int(minx), int(maxx) + 1, int(miny), int(maxy) + 1

        new_box_sum = -1
        loop_cnt = 0
        while new_box_sum != 0 and loop_cnt < max_loop:

            if minx > 100 and miny > 166:
                sx = np.random.randint(110, 250)
                sy = np.random.randint(170, 280)
            elif maxx < 100 and miny > 166:
                sx = np.random.randint(10, 80)
                sy = np.random.randint(170, 280)
            elif maxx < 100 and maxy < 166:
                sx = np.random.randint(10, 80)
                sy = np.random.randint(10, 150)
            elif minx > 100 and maxy < 166:
                sx = np.random.randint(110, 250)
                sy = np.random.randint(10, 150)
            else:
                sx = minx
                sy = miny

            if sx + maxx - minx > 300 or sy + maxy - miny > 330:
                loop_cnt += 1
                new_box_sum = -1
                continue
            new_box = gt_seg[sx:sx + maxx - minx, sy:sy + maxy - miny]
            new_box_sum = np.sum(new_box)
            loop_cnt += 1
        if loop_cnt >= max_loop:
            return points, gt_seg, False
        gt_seg[sx:sx + maxx - minx, sy:sy + maxy - miny] = extra_gt_seg[minx:maxx, miny:maxy]

        range_minx = minx * voxel_size[0] + AREA_EXTENTS[0][0]
        range_maxx = maxx * voxel_size[0] + AREA_EXTENTS[0][0]
        range_miny = miny * voxel_size[1] + AREA_EXTENTS[1][0]
        range_maxy = maxy * voxel_size[1] + AREA_EXTENTS[1][0]
        range_sx = sx * voxel_size[0] + AREA_EXTENTS[0][0]
        range_sy = sy * voxel_size[1] + AREA_EXTENTS[1][0]



        repeat_points_mask1 = (range_minx <= extra_points[:, 0]) & (extra_points[:, 0] <= range_maxx)
        repeat_points_mask2 = (range_miny <= extra_points[:, 1]) & (extra_points[:, 1] <= range_maxy)
        repeat_points_mask = repeat_points_mask1 & repeat_points_mask2
        repeat_points = extra_points[repeat_points_mask]
        repeat_points[:, 0] += (range_sx - range_minx)
        repeat_points[:, 1] += (range_sy - range_miny)
        points = np.concatenate([points, repeat_points], axis=0)

    return points, gt_seg, True

class PC_BEV_Dataset(torch.utils.data.Dataset):
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
        img, target = self.get_ele(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def order_img_list(self):
        img_list = []
        for key, ele in self.imgs.items():
            img_list.extend(ele)
        shuffle(img_list)
        self.imgs = img_list
        print('lenth of img_list:', len(self.imgs))

    def get_ele(self, idx):
        img_path = self.imgs[idx]

        if img_path.endswith('.txt'):
            points = np.loadtxt(img_path)
        else:
            points = read_pcd(img_path)

        label_path = img_path.replace('images', 'annotations')#('images_fine', 'annotations0503_fine')
        if label_path.endswith('.pcd'):
            label_path = label_path.replace(".pcd", ".txt")
        elif label_path.endswith('.bin'):
            label_path = label_path.replace(".bin", ".txt")

        gt_info = np.loadtxt(label_path, dtype=str)
        gt_semantic_seg = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.int32)
        length = len(gt_info)
        for i in range(length):
            x = int(gt_info[i][0])
            y = int(gt_info[i][1])

            if self.config['multiple_classes']:
                cate = gt_info[i][2].lower()

                assert cate in CATEGORY_TO_ID.keys()

                cate_id = CATEGORY_TO_ID[cate]
                gt_semantic_seg[x, y, 0] = cate_id
            else:
                if x < DETECTION_AREA_EXTENTS[0][0] or x >= DETECTION_AREA_EXTENTS[0][1] or y < DETECTION_AREA_EXTENTS[1][0] or y >= DETECTION_AREA_EXTENTS[1][1]:
                    continue
                else:
                    x = x - DETECTION_AREA_EXTENTS[0][0]
                    y = y - DETECTION_AREA_EXTENTS[1][0]
                is_obstacle = int(gt_info[i][2])
                is_valid = int(gt_info[i][4])
                if is_obstacle == 1 and is_valid == 0:
                    gt_semantic_seg[x, y, 0] = 1

        target = gt_semantic_seg.squeeze().astype(np.uint8)

        extract_bev = ExtractBevFeature(area_extents=AREA_EXTENTS,
                                        self_car_extents=SELF_CAR_EXTENTS,
                                        num_slices=num_slice,
                                        voxel_size=voxel_size,
                                        log_norm=log_norm)
        bev_img = extract_bev(points)

        img = bev_img

        return img, target

    def __len__(self):
        return len(self.imgs)