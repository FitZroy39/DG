import os.path
import os.path as osp
import cv2
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import open3d as o3d
import warnings

BEV_FACTOR = 2
Y_MIN = -45
Y_MAX = 45
X_MIN = -50
X_MAX = 100
VOXEL_X_SIZE = 0.1
VOXEL_Y_SIZE = 0.1
RADIUS_1 = int(40 / VOXEL_X_SIZE * BEV_FACTOR)
RADIUS_2 = int(70 / VOXEL_X_SIZE * BEV_FACTOR)
IMG_WIDTH = int((X_MAX - X_MIN) / VOXEL_X_SIZE)
IMG_HEIGHT = int((Y_MAX - Y_MIN) / VOXEL_Y_SIZE)
CENTER_X = int((-X_MIN) / VOXEL_X_SIZE) * BEV_FACTOR
CENTER_Y = int((-Y_MIN) / VOXEL_Y_SIZE) * BEV_FACTOR

GRID_X = 0.5
GRID_Y = 0.3

gird_x_range = np.arange(X_MIN, X_MAX, GRID_X)
gird_y_range = np.arange(Y_MIN, Y_MAX, GRID_Y)
grid_map_x, grid_map_y = np.meshgrid(gird_x_range, gird_y_range)

factor_x = 1 / VOXEL_X_SIZE * BEV_FACTOR
factor_y = 1 / VOXEL_Y_SIZE * BEV_FACTOR

@njit(cache=True, fastmath=True)
def draw_point_cloud_jit(points, birdview):
    factor_x = 1 / VOXEL_X_SIZE * BEV_FACTOR
    factor_y = 1 / VOXEL_Y_SIZE * BEV_FACTOR
    length = len(points)
    for i in range(length):
        x = points[i][0]
        y = points[i][1]
        if X_MIN < x < X_MAX:
            if Y_MIN < y < Y_MAX:
                x_b = int((x - X_MIN) * factor_x)
                y_b = int((-y - Y_MIN) * factor_y)
                birdview[y_b, x_b, 0] = 255
                birdview[y_b, x_b, 1] = 255
                birdview[y_b, x_b, 2] = 255
    return birdview

def read_pcd(file_path):
    if file_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    elif file_path.endswith('.bin'):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

def draw_bg(pcd_path):
    """
    background for bev includes point clouds, hdmap mask, reference distance,
    and color scheme for boxes
    """
    canvas = np.zeros(
        (IMG_HEIGHT * BEV_FACTOR, IMG_WIDTH * BEV_FACTOR, 3), dtype=np.float32
    )
    points = read_pcd(pcd_path)
    bev = draw_point_cloud_jit(points, canvas)

    # draw reference distance
    center_x = CENTER_X
    center_y = CENTER_Y
    bev = bev.astype(np.float32)
    cv2.circle(bev, (center_x, center_y), 4, (255, 255, 0), 1)
    cv2.circle(bev, (center_x, center_y), RADIUS_1, (255, 255, 0), 1)
    cv2.circle(bev, (center_x, center_y), RADIUS_2, (255, 255, 0), 1)

    return bev

def trans_coord_lidar_to_bev(x, y):
    """
    convert ego coordinates to bev coordinates.
    bev: x-positive to the right, y-positive to up
    """
    a = (x - X_MIN) / VOXEL_X_SIZE * BEV_FACTOR
    b = (-y - Y_MIN) / VOXEL_Y_SIZE * BEV_FACTOR
    a = np.clip(a, a_max=(X_MAX - X_MIN) / VOXEL_X_SIZE * BEV_FACTOR, a_min=0)
    b = np.clip(b, a_max=(Y_MAX - Y_MIN) / VOXEL_Y_SIZE * BEV_FACTOR, a_min=0)
    return a, b


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


def draw_rect(bev, rect_points, color, line_width=2):
    cv2.line(
        bev,
        (rect_points[0, 0], rect_points[0, 1]),
        (rect_points[1, 0], rect_points[1, 1]),
        color,
        line_width,
        cv2.LINE_AA,
    )
    cv2.line(
        bev,
        (rect_points[1, 0], rect_points[1, 1]),
        (rect_points[2, 0], rect_points[2, 1]),
        color,
        line_width,
        cv2.LINE_AA,
    )
    cv2.line(
        bev,
        (rect_points[2, 0], rect_points[2, 1]),
        (rect_points[3, 0], rect_points[3, 1]),
        color,
        line_width,
        cv2.LINE_AA,
    )
    cv2.line(
        bev,
        (rect_points[3, 0], rect_points[3, 1]),
        (rect_points[0, 0], rect_points[0, 1]),
        color,
        line_width,
        cv2.LINE_AA,
    )
    return bev

def draw_rotate_rect(
    bev, rect, texts=[], color=(0, 0, 128), font_size=0.6, font_weight=2, line_width=2
):
    rect_points = rotate_rect(rect)
    for i in range(len(rect_points)):
        img_x, img_y = trans_coord_lidar_to_bev(rect_points[i, 0], rect_points[i, 1])
        rect_points[i, 0] = int(img_x)
        rect_points[i, 1] = int(img_y)
    rect_points = rect_points.astype(np.int32)
    bev = draw_rect(bev, rect_points, color, line_width)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(texts)):
        cv2.putText(
            bev,
            texts[i],
            (rect_points[1, 0] + 2 + 80 * i, rect_points[1, 1] - 8),
            font,
            font_size,
            color,
            font_weight,
        )
    return bev


def vis_single_result(pcd_path, save_path, gt_path, pedestrian_list):
    plt.figure("gt",dpi=200)#, figsize=(20,24))


    bev2 = draw_bg(pcd_path)
    gt_data = np.loadtxt(gt_path, dtype=np.int32)#[:,:2]
    gt_grids = gt_data[:,:2]
    is_obstacle = gt_data[:, 2]
    is_valid = gt_data[:, 4]


    for i, grid in enumerate(gt_grids):
        idx = int(grid[0])
        idy = int(grid[1])
        if is_obstacle[i] == 0 or is_valid[i] == 1:
            continue

        img2 = draw_rotate_rect(
            bev2,
            rect=[grid_map_x[idy][idx] + GRID_X / 2, grid_map_y[idy][idx] + GRID_Y / 2, GRID_X, GRID_Y, 0],
            texts=[],
            color=(239, 255, 0),
            line_width=1,
            font_size=0.3
        )


    for grid in pedestrian_list:
        idx = int(grid[0])
        idy = int(grid[1])
        bev3 = draw_rotate_rect(
            img2,
            rect = [idx, idy, GRID_X, GRID_Y, 0],
            texts=[],
            color=(0, 0, 255),
            line_width=1,
            font_size=0.3
        )

    plt.subplot(1, 1, 1), plt.title('gt')
    plt.imshow(img2.astype(np.uint8))


    save_base = os.path.join(save_path, gt_path.split('/')[-3])
    save_name = os.path.join(save_base, gt_path.split('/')[-1].replace('.txt', '.jpg'))
    if not os.path.exists(save_base):
        os.makedirs(save_base)
    plt.savefig(save_name)
    plt.clf()

def get_pedestrian(obstacle_path):
    pedestrian_list = [] 
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
        # out of range
        if obstacle_bounding_box_ego_x[i] < -5 or obstacle_bounding_box_ego_x[i] > 90 or obstacle_bounding_box_ego_y[i] < -30 or obstacle_bounding_box_ego_y[i] > 30:
            continue
        if int(obstacle_bounding_box_type[i]) == 3:
            pedestrian_list.append([obstacle_bounding_box_ego_x[i], obstacle_bounding_box_ego_y[i]])
    return pedestrian_list

if __name__ == "__main__":

    obstacle_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/pedestrain/obstacle"
    gt_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/pedestrain/gt"
    pcd_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/pedestrain/pcd"
    save_path = '/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/8vis'
    
    gt_dir_list = sorted(os.listdir(gt_dir))

    for bag_name in gt_dir_list:
        obstacle_list = sorted(os.listdir(osp.join(obstacle_dir, bag_name)))
        gt_list = sorted(os.listdir(osp.join(gt_dir, bag_name)))

        for gt_file in gt_list: 
            gt_unify_time_path = gt_file[:14] + '.txt'

            obstacle_path = osp.join(obstacle_dir, bag_name, gt_unify_time_path)
            gt_path = osp.join(gt_dir, bag_name, gt_file)

            if gt_unify_time_path not in obstacle_list:
                print("Not find file in obstacle detection {}".format(gt_file))
                continue
            
            pedestrian_list = get_pedestrian(obstacle_path)
            
            if len(pedestrian_list) == 0:
                continue
            pcd_path = osp.join(pcd_dir, bag_name, gt_file.replace('.txt', '.pcd'))
            vis_single_result(pcd_path, save_path, gt_path, pedestrian_list)
            