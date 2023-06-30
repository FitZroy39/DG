import os.path
import os.path as osp
import cv2
import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
BEV_FACTOR = 2
# Y_MIN = -45
# Y_MAX = 45
# X_MIN = -50
# X_MAX = 100
Y_MIN = -40.5
Y_MAX = 40.5
X_MIN = -25
X_MAX = 60
# - 5 90 , -30,30
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

def vis_single_result(pcd_path, save_path, pred_path, gt_path):

    if isinstance(pred_path, str):
        res = np.loadtxt(pred_path, dtype=np.int32)
        # grids = [tuple([g[0]- DETECTION_AREA_EXTENTS[0][0], g[1]- DETECTION_AREA_EXTENTS[1][0]]) for g in res if g[2] == 1]
        grids = [tuple([g[0], g[1]]) for g in res if g[2] == 1]

    bev2 = draw_bg(pcd_path)
    gt_data = np.loadtxt(gt_path, dtype=np.int32)
    grids = [tuple(g) for g in grids]
    grids_set = set(tuple(grids))
    
    gt_grids = []
    for gg in gt_data:
        # if gg[2] == 0 and gg[4] == 1:
        if gg[2] == 0:
            continue

        # if gg[0] < DETECTION_AREA_EXTENTS[0][0] or gg[0] >= DETECTION_AREA_EXTENTS[0][1] or gg[1] < DETECTION_AREA_EXTENTS[1][0] or gg[1] >= DETECTION_AREA_EXTENTS[1][1]:
        #     continue
        # else:
        #     idx = gg[0] - DETECTION_AREA_EXTENTS[0][0]
        #     idy = gg[1] - DETECTION_AREA_EXTENTS[1][0]
        #     gt_grids.append((idx, idy))
        gt_grids.append((gg[0], gg[1]))

    gt_grids_set = set(tuple(gt_grids))
    intersect_set = grids_set & gt_grids_set
    missed_detect = gt_grids_set - intersect_set
    false_detect = grids_set - intersect_set

    for grid in gt_grids:
        idx = int(grid[0])
        idy = int(grid[1])

        img2 = draw_rotate_rect(
            bev2,
            rect=[grid_map_x[idy][idx] + GRID_X/2, grid_map_y[idy][idx] + GRID_Y/2, GRID_X, GRID_Y, 0],
            texts=[],
            color=(239, 255, 0),
            line_width=1,
            font_size=0.3
        )
    missed_detect = np.array(list(missed_detect))
    for grid in missed_detect:
        idx = int(grid[0])
        idy = int(grid[1])
        # cnt_z = str(int(grid[2]))

        img2 = draw_rotate_rect(
            img2,
            rect=[grid_map_x[idy][idx] + GRID_X/2, grid_map_y[idy][idx] + GRID_Y/2, GRID_X, GRID_Y, 0],
            texts=[],
            color=(0, 0, 255),
            line_width=1,
            font_size=0.3
        )
    false_detect = np.array(list(false_detect))
    for grid in false_detect:
        idx = int(grid[0])
        idy = int(grid[1])
        # cnt_z = str(int(grid[2]))

        img2 = draw_rotate_rect(
            img2,
            rect=[grid_map_x[idy][idx] + GRID_X/2, grid_map_y[idy][idx] + GRID_Y/2, GRID_X, GRID_Y, 0],
            texts=[],
            color=(0, 255, 0),
            line_width=1,
            font_size=0.3
        )

    BOX_TYPES = ['correct detect', 'missed detect', 'false detect']
    BOX_TYPE_COLORS = [(239, 255, 0), (0, 0, 255), (0, 255, 0)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    rect_w = 1.0 / 8 * IMG_WIDTH * BEV_FACTOR
    rect_h = 1.0 / 4 * rect_w
    delta = 7
    rect_x = IMG_WIDTH * BEV_FACTOR - rect_w - 10
    rect_y = delta
    font_size = 0.8
    font_weight = 2
    for i in range(len(BOX_TYPES)):
        category = BOX_TYPES[i]
        type_color = BOX_TYPE_COLORS[i]
        cv2.rectangle(
            img2,
            (int(rect_x), int(rect_y + i * rect_h + delta)),
            (int(rect_x + rect_w), int(rect_y + i * rect_h + rect_h)),
            type_color,
            thickness=-1,
        )
        cv2.putText(
            img2,
            category,
            (
                int(rect_x + rect_w / 10),
                int(rect_y + i * rect_h + delta + rect_h * 0.6),
            ),
            font,
            font_size,
            (255, 255, 255),
            font_weight,
        )

    save_name = os.path.join(save_path, gt_path.split('/')[-1].replace('.txt', '.jpg'))
    cv2.imwrite(save_name, img2)

if __name__ == "__main__":
    # bag_root = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/linyuqi/grid_benchmark0503_300x300/grid_benchmark0503_300x300_val"
    bag_root = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/ray_gt"
    save_path = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/vis_raycom2"
    bag_list = sorted(os.listdir(bag_root))[5:6]
    print('{} bags to vis...'.format(len(bag_list)))

    for bag in tqdm(bag_list):
        print('>>>processing {}'.format(bag))
        pcd_path = os.path.join(bag_root, bag, 'pcd')
        gt_path = os.path.join(bag_root, bag, 'gt')
        baseline_path = os.path.join(bag_root, bag, 'baseline_gt')
        pred_path = os.path.join("/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/ray_val", bag)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        bags = sorted(os.listdir(gt_path)) 
        for file in bags:
            pcd_name = os.path.join(pcd_path, file.replace('.txt', '.pcd'))
            if not os.path.exists(pcd_name):
                continue
            gt_name = os.path.join(gt_path, file)
            pred_name = os.path.join(pred_path, 'gt', file)
            baseline_name = os.path.join(baseline_path, file)
            # print("pred_name: ", pred_name)
            if not os.path.exists(pred_name):
                continue
            # if not os.path.exists(pred_name) or not os.path.exists(baseline_name):
            #     continue
            # print("hahahahhaha")
            vis_single_result(pcd_name, save_path, pred_name, gt_name)