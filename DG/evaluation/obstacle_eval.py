import os
import os.path as osp
import cv2
import numpy as np
from metrics import intersect_and_union_with_mask
from prettytable import PrettyTable
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


number_of_obstacles_with_range = {}
def get_number_of_obstacles_with_range_dict(obstacle_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obstacle_bounding_box_list = np.loadtxt(obstacle_path, dtype=np.float64)

    if len(obstacle_bounding_box_list) == 0:
        return

    # check dimention of list
    obstacle_bounding_box_list = np.array(obstacle_bounding_box_list)
    if len(obstacle_bounding_box_list.shape) == 1:
        obstacle_bounding_box_list = obstacle_bounding_box_list.reshape(-1, len(obstacle_bounding_box_list))

    obstacle_bounding_box_type = obstacle_bounding_box_list[:, 0]
    obstacle_bounding_box_ego_x = obstacle_bounding_box_list[:, 1]
    obstacle_bounding_box_ego_y = obstacle_bounding_box_list[:, 2]
    for i in range(len(obstacle_bounding_box_list)):
        if obstacle_bounding_box_ego_x[i] < AREA_EXTENTS[0][0] or obstacle_bounding_box_ego_x[i] > AREA_EXTENTS[0][1] or obstacle_bounding_box_ego_y[i] < AREA_EXTENTS[1][0] or obstacle_bounding_box_ego_y[i] > AREA_EXTENTS[1][1]:
            continue
        if obstacle_bounding_box_type[i] not in number_of_obstacles_with_range:
            number_of_obstacles_with_range[int(obstacle_bounding_box_type[i])] = [0, 0]
        if obstacle_bounding_box_ego_x[i] > 30:
            number_of_obstacles_with_range[int(obstacle_bounding_box_type[i])][1] += 1
        else:
            number_of_obstacles_with_range[int(obstacle_bounding_box_type[i])][0] += 1


# every file
def get_bounding_box_with_point_list(obstacle_path, gt_path):


    exclude_bounding_box_set = []
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
        exclude_bounding_box_set = [grid for i, grid in enumerate(gt_grids)]
        return [], [], {}, exclude_bounding_box_set


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
           
                
    # check is in range
    is_obstacle_in_range = []
    for i in range(len(obstacle_bounding_box_list)):
        if obstacle_bounding_box_ego_x[i] > 30:
            is_obstacle_in_range.append(False)
        else:
            is_obstacle_in_range.append(True)
        



    # traverse point
    for i, grid in enumerate(gt_grids):
        point_grid_idx = int(grid[0])
        point_grid_idy = int(grid[1])
        
        if gt_is_obstacle[i] == 0 or gt_is_invalid[i] == 1:
            continue
        if point_grid_idx < DETECTION_AREA_EXTENTS[0][0] or point_grid_idx >= DETECTION_AREA_EXTENTS[0][1] or point_grid_idy < DETECTION_AREA_EXTENTS[1][0] or point_grid_idy >= DETECTION_AREA_EXTENTS[1][1]:
            continue
        point_ego_x, point_ego_y = trans_grid_to_ego(point_grid_idx, point_grid_idy)
        
        is_exclude_bounding_box_set = True

        # traverse obstacle bounding box
        for j in range(len(obstacle_bounding_box_list)):
            if is_point_in_Rectangle(point_ego_x, point_ego_y, obstacle_bounding_box_ego_x[j], obstacle_bounding_box_ego_y[j], obstacle_bounding_box_length[j], obstacle_bounding_box_width[j], obstacle_bounding_box_theta[j]):
                is_exclude_bounding_box_set = False
                
                if j in bounding_box_set:
                    bounding_box_set[j].append(grid)
                else:
                    bounding_box_set[j] = [grid]
                
                break
            
            # truck have head
            if obstacle_bounding_box_type[j] == 7:
                if is_point_in_Rectangle(point_ego_x, point_ego_y, obstacle_bounding_box_head_ego_x[j], obstacle_bounding_box_head_ego_y[j], obstacle_bounding_box_head_length[j], obstacle_bounding_box_head_width[j], obstacle_bounding_box_head_theta[j]):
                    is_exclude_bounding_box_set = False
                    if j in bounding_box_set:
                        bounding_box_set[j].append(grid)
                    else:
                        bounding_box_set[j] = [grid]  

                    break


        if is_exclude_bounding_box_set:
            exclude_bounding_box_set.append(grid)
            
    

    return obstacle_bounding_box_type, is_obstacle_in_range, bounding_box_set, exclude_bounding_box_set


def load_and_transform(data):
    bev_x_h = round((AREA_EXTENTS[0][1] - AREA_EXTENTS[0][0]) / voxel_size[0])
    bev_y_w = round((AREA_EXTENTS[1][1] - AREA_EXTENTS[1][0]) / voxel_size[1])
    semantic_seg = np.zeros((bev_x_h, bev_y_w, 1), dtype=np.int32)
    for i in range(len(data)):
        x = data[i][0]
        y = data[i][1]
        if x < DETECTION_AREA_EXTENTS[0][0] or x >= DETECTION_AREA_EXTENTS[0][1] or y < DETECTION_AREA_EXTENTS[1][0] or y >= DETECTION_AREA_EXTENTS[1][1]:
            continue
        else:
            x = x - DETECTION_AREA_EXTENTS[0][0]
            y = y - DETECTION_AREA_EXTENTS[1][0]

        semantic_seg[x, y, 0] = 1
    seg_map = semantic_seg.squeeze().astype(np.uint8)
    return seg_map

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

def ignore_occluded_grid(grids, tp_grids, mask):
    for grid in grids:
        idx = int(grid[0])
        idy = int(grid[1])
        px = (idx - MIN_X) * voxel_size[0] + voxel_size[0] / 2
        py = (idy - MIN_Y) * voxel_size[1] + voxel_size[1] / 2
        for tp_grid in tp_grids:
            tp_idx = int(tp_grid[0])
            tp_idy = int(tp_grid[1])

            x1 = (tp_idx - MIN_X) * voxel_size[0]
            y1 = (tp_idy - MIN_Y) * voxel_size[1] + voxel_size[1]
            x2 = (tp_idx - MIN_X) * voxel_size[0] + voxel_size[0]
            y2 = (tp_idy - MIN_Y) * voxel_size[1]
            first_quadrant_valid = (px > 0 and py > 0 and x1 < px and y2 < py)
            second_quadrant_valid = (px < 0 and py > 0 and x2 > px and y2 < py)
            third_quadrant_valid = (px < 0 and py < 0 and x2 > px and y1 > py)
            fourth_quadrant_valid = (px > 0 and py < 0 and x1 < px and y1 > py)
            if first_quadrant_valid or second_quadrant_valid or third_quadrant_valid or fourth_quadrant_valid:
                is_occluded = is_line_intersect_gird(0,0,px,py,x1,y1,x2,y2)
                if is_occluded:
                    mask[idx, idy] = 0
                    break
    return

def generate_obs_mask2(res, gt_seg):

    grids = []
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if res[i][j] == 1:
                grids.append([i, j])
    grids = [tuple(g) for g in grids]
    grids_set = set(tuple(grids))
    gt_grids = []
    for i in range(gt_seg.shape[0]):
        for j in range(gt_seg.shape[1]):
            if gt_seg[i][j] == 1:
                gt_grids.append([i, j])
    gt_grids = [tuple(gg) for gg in gt_grids]
    gt_grids_set = set(tuple(gt_grids))
    intersect_set = grids_set & gt_grids_set

    missed_detect = gt_grids_set - intersect_set
    false_detect = grids_set - intersect_set
    intersect_set = np.array(list(intersect_set))
    missed_detect = np.array(list(missed_detect))
    false_detect = np.array(list(false_detect))
    bev_x_h = round((AREA_EXTENTS[0][1] - AREA_EXTENTS[0][0]) / voxel_size[0])
    bev_y_w = round((AREA_EXTENTS[1][1] - AREA_EXTENTS[1][0]) / voxel_size[1])
    mask = np.ones((bev_x_h, bev_y_w))

    ignore_occluded_grid(missed_detect, intersect_set, mask)
    ignore_occluded_grid(false_detect, intersect_set, mask)
    return mask


def get_recall_in_bounding_box(gt_point_list, pred_point_list):
    pre_eval_results = []
    gt_seg_map = load_and_transform(gt_point_list)
    pred_seg_map = load_and_transform(pred_point_list)

    # with occlusion mask
    mask = generate_obs_mask2(pred_seg_map, gt_seg_map)
    pre_eval_results.append(intersect_and_union_with_mask(pred_seg_map, gt_seg_map, 2, 255, dict(),
                                    False, mask>0))

    # without occlusion mask
    # pre_eval_results.append(intersect_and_union(pred_seg_map, gt_seg_map, 2, 255, dict(), False))

    pre_eval_results = tuple(zip(*pre_eval_results))
    total_area_intersect = sum(pre_eval_results[0])
    total_area_label = sum(pre_eval_results[3])
    recall = total_area_intersect / total_area_label
    return recall[-1].numpy()



def get_obstacle_eval_result(obstacle_dir, pred_gt_dir, gt_dir):

    gt_dir_list = sorted(os.listdir(gt_dir))
    recall_lower_bound = 0.8

    recall_of_obstacles_with_range = {}

    num_pedestrain = 0


    for bag_name in gt_dir_list:
        obstacle_list = sorted(os.listdir(osp.join(obstacle_dir, bag_name)))
        pred_gt_list = sorted(os.listdir(osp.join(pred_gt_dir, bag_name)))

        for pred_gt_file in pred_gt_list: 
            pred_gt_unify_time_path = pred_gt_file[:14] + '.txt'

            obstacle_path = osp.join(obstacle_dir, bag_name, pred_gt_unify_time_path)
            pred_gt_path = osp.join(pred_gt_dir, bag_name, pred_gt_file)
            gt_path = osp.join(gt_dir, bag_name, pred_gt_file)

            if pred_gt_unify_time_path not in obstacle_list:
                print("Not find file in obstacle detection {}".format(pred_gt_file))
                continue

            
            obstacle_bounding_box_type, is_obstacle_in_range, gt_bounding_box_with_point_list, _ = get_bounding_box_with_point_list(obstacle_path, gt_path)
            _, _, pred_gt_bounding_box_with_point_list, _ = get_bounding_box_with_point_list(obstacle_path, pred_gt_path)
            get_number_of_obstacles_with_range_dict(obstacle_path)
            

            for bounding_box_index in gt_bounding_box_with_point_list:
                idx = int(obstacle_bounding_box_type[bounding_box_index])
                
                if bounding_box_index not in pred_gt_bounding_box_with_point_list.keys():    
                    continue
                
                recall = get_recall_in_bounding_box(gt_bounding_box_with_point_list[bounding_box_index], pred_gt_bounding_box_with_point_list[bounding_box_index])

                if idx not in recall_of_obstacles_with_range.keys():
                    recall_of_obstacles_with_range[idx] = [0]*4
                if idx == 3:
                    num_pedestrain += 1
                if is_obstacle_in_range[bounding_box_index]:
                    recall_of_obstacles_with_range[idx][1] += 1
                    if recall > recall_lower_bound:
                        recall_of_obstacles_with_range[idx][0] += 1
                    elif idx == 3:
                        print(pred_gt_unify_time_path)
                else:
                    recall_of_obstacles_with_range[idx][3] += 1
                    if recall > recall_lower_bound:
                        recall_of_obstacles_with_range[idx][2] += 1



    print("--------NUMBER--------")
    number_table_data = PrettyTable()
    number_table_data.field_names = ['Type', 'x<30', 'x>30']
    for key, val in number_of_obstacles_with_range.items():
        number_table_data.add_row([key, val[0], val[1]])
    print(number_table_data)

    print("--------RECALL--------")
    recall_table_data = PrettyTable()
    recall_table_data.field_names = ['Type', 'x<30', 'x>30']
    for key, val in recall_of_obstacles_with_range.items():
        in_range_recall = 'N/A'
        out_range_recall = 'N/A'
        if val[1] != 0:
            in_range_recall = round(val[0]/val[1], 3)
        if val[3] != 0:
            out_range_recall = round(val[2]/val[3], 3)
        recall_table_data.add_row([key, in_range_recall, out_range_recall])
    print(recall_table_data)
    print("num_pedestrain: ", num_pedestrain)



if __name__ == "__main__":

    obstacle_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/deep-lidar-model/preds/20w_300x300_final2_obstacle"

    pred_gt_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/20w_170x270_eval"
    gt_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/eva/gt"

    get_obstacle_eval_result(obstacle_dir, pred_gt_dir, gt_dir)