import os
import os.path as osp
import cv2
import numpy as np
import warnings
import sys
sys.path.append("..") 
from configs import AREA_EXTENTS,voxel_size,DETECTION_AREA_EXTENTS


MIN_X = int(abs(AREA_EXTENTS[0][0]/voxel_size[0]))
MIN_Y = int(abs(AREA_EXTENTS[1][0]/voxel_size[1]))

def trans_grid_to_ego(coord_x, coord_y):
    offset_x = -50
    offset_y = -45
    scale_x = 0.5
    scale_y = 0.3
    
    # tran2ego
    corner_x = offset_x + coord_x * scale_x
    corner_y = offset_y + coord_y * scale_y
    center_x = corner_x + 0.5 * scale_x
    center_y = corner_y + 0.5 * scale_y

    return center_x, center_y

def rotate_rect(rect):
    x, y, l, w, theta = rect
    rect_points = np.array(
        [[-l / 2, -l / 2, l / 2, l / 2], [w / 2, -w / 2, -w / 2, w / 2]]
    )
    rot_mat = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    rect_points = np.dot(rot_mat, rect_points) + np.tile([x, y], (4, 1)).T
    return rect_points.transpose()

def is_point_in_Rectangle(point_x, point_y, bounding_box_x, bounding_box_y, bounding_box_length, bounding_box_width, theta):
    
    point = (point_x, point_y)
    length_bias = 0.2
    width_bias = 0.15

    rect = [bounding_box_x, bounding_box_y, bounding_box_length+length_bias, bounding_box_width+width_bias, theta]
    bounding_box_rect_corners = np.float32(rotate_rect(rect).reshape(-1, 1, 2))
    
    result = cv2.pointPolygonTest(bounding_box_rect_corners, point, False)

    # result: 1.0 inner 0 on -1.0 outter
    if result >= 0:
        return True
    else:
        return False

def get_bounding_box_with_point_list(obstacle_path, gt_path):

    bounding_box_set = {}

    gt_path_list = np.loadtxt(gt_path, dtype=np.int32)
    gt_grids = gt_path_list[:, :2]
    gt_is_obstacle = gt_path_list[:, 2]
    gt_is_invalid = gt_path_list[:, 4]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obstacle_bounding_box_list = np.loadtxt(obstacle_path, dtype=np.float64)

    # zero obstacle bounding box
    if len(obstacle_bounding_box_list) == 0:
        return [], [], {}, []


    # check dimention of list
    obstacle_bounding_box_list = np.array(obstacle_bounding_box_list)
    if len(obstacle_bounding_box_list.shape) == 1:
        obstacle_bounding_box_list = obstacle_bounding_box_list.reshape(-1, len(obstacle_bounding_box_list))



    obstacle_bounding_box_type = obstacle_bounding_box_list[:, 0]
    obstacle_bounding_box_ego_x = obstacle_bounding_box_list[:, 1]
    obstacle_bounding_box_ego_y = obstacle_bounding_box_list[:, 2]
    obstacle_bounding_box_length = obstacle_bounding_box_list[:, 3]
    obstacle_bounding_box_width = obstacle_bounding_box_list[:, 4]
    obstacle_bounding_box_theta = obstacle_bounding_box_list[:, 5]

    obstacle_bounding_box_head_ego_x = obstacle_bounding_box_list[:, 6]
    obstacle_bounding_box_head_ego_y = obstacle_bounding_box_list[:, 7]
    obstacle_bounding_box_head_length = obstacle_bounding_box_list[:, 8]
    obstacle_bounding_box_head_width = obstacle_bounding_box_list[:, 9]
    obstacle_bounding_box_head_theta = obstacle_bounding_box_list[:, 10]
        


    # traverse point
    for i, grid in enumerate(gt_grids):
        point_grid_idx = int(grid[0])
        point_grid_idy = int(grid[1])
        
        if gt_is_obstacle[i] == 0 or gt_is_invalid[i] == 1:
            continue
        if point_grid_idx < DETECTION_AREA_EXTENTS[0][0] or point_grid_idx >= DETECTION_AREA_EXTENTS[0][1] or point_grid_idy < DETECTION_AREA_EXTENTS[1][0] or point_grid_idy >= DETECTION_AREA_EXTENTS[1][1]:
            continue
        point_ego_x, point_ego_y = trans_grid_to_ego(point_grid_idx, point_grid_idy)
        

        # traverse obstacle bounding box
        for j in range(len(obstacle_bounding_box_list)):
            if is_point_in_Rectangle(point_ego_x, point_ego_y, obstacle_bounding_box_ego_x[j], obstacle_bounding_box_ego_y[j], obstacle_bounding_box_length[j], obstacle_bounding_box_width[j], obstacle_bounding_box_theta[j]):
             
                
                if j in bounding_box_set:
                    bounding_box_set[j].append(grid)
                else:
                    bounding_box_set[j] = [grid]
                
                break
            
            # truck have head
            if obstacle_bounding_box_type[j] == 7:
                if is_point_in_Rectangle(point_ego_x, point_ego_y, obstacle_bounding_box_head_ego_x[j], obstacle_bounding_box_head_ego_y[j], obstacle_bounding_box_head_length[j], obstacle_bounding_box_head_width[j], obstacle_bounding_box_head_theta[j]):
                
                    if j in bounding_box_set:
                        bounding_box_set[j].append(grid)
                    else:
                        bounding_box_set[j] = [grid]  

                    break
    
    obstacles_grid_list = []
    for bounding_box_index in bounding_box_set:
        idx = int(obstacle_bounding_box_type[bounding_box_index])
        for grid in bounding_box_set[bounding_box_index]:
            obstacles_grid = "{} {} {}".format(grid[0], grid[1], idx)
            obstacles_grid_list.append(obstacles_grid)



    return obstacles_grid_list

if __name__ == "__main__":

    # obstacle_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/deep-lidar-model/preds/20w_300x300_final2_obstacle"
    # gt_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/eva/gt"
    # save_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/obstacle_grid"

    obstacle_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/howo_v4_obstacle"
    gt_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/howo_v4"
    save_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/howo_v4_mul"
        

    gt_dir_list = sorted(os.listdir(gt_dir))
    for bag_name in gt_dir_list:
        obstacle_list = sorted(os.listdir(osp.join(obstacle_dir, bag_name)))
        gt_list = sorted(os.listdir(osp.join(gt_dir, bag_name, "gt")))
        # gt_list = sorted(os.listdir(osp.join(gt_dir, bag_name)))

        save_bag_dir = osp.join(save_dir, bag_name)
        if not os.path.exists(save_bag_dir):
            os.makedirs(save_bag_dir)

        for gt_file in gt_list: 
            gt_unify_time_path = gt_file[:14] + '.txt'   # use for howo
            # gt_unify_time_path = gt_file[:12] + '.txt'

            obstacle_path = osp.join(obstacle_dir, bag_name, gt_unify_time_path)
            gt_path = osp.join(gt_dir, bag_name, 'gt', gt_file)

            if gt_unify_time_path not in obstacle_list:
                print("Not find file in obstacle detection {}".format(gt_file))
                continue
            
            obstacles_grid_list = get_bounding_box_with_point_list(obstacle_path, gt_path)

            msg_path = osp.join(save_bag_dir, gt_file)
            with open(msg_path, 'w') as fi:
                for li in obstacles_grid_list:
                    fi.writelines("{}\n".format(li))
        print("finish {}".format(bag_name))
