import os
import os.path
import os.path as osp
import cv2
import numpy as np
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
from lidar_frame_label_pb2 import LidarFrameLabel
from google.protobuf import text_format
import pickle
from tqdm import tqdm

BIG_OBJECT_TYPES = [
    "car",
    "truck",
    "tricycle",
    "forklift",
    "forkliftbox",
    "gantry",
    "crane",
    #"selftrailer",
    "otherbigvehicle",
    "stacker",
    "scrapper",
    "hatchway",
    "container",
    "static",
    "static_group",
    "lockbox",
    "heavytrucktrailer",
    "heavytruckhead",
    "peggingblock",
]
SMALL_OBJECT_TYPES = [
    "pedestrian",
    "cyclist",
    "cylindarblock",
    "coneblock",
    "gocart",
    "directionpost",
    "fans",
    "flag",
]
NOISE_TYPES = [
    "noise",
    "rainynoise",
    "spraynoise",
    "sandynoise",
    "foggynoise",
    "othernoise",
    "groundbesidetree",
]


CATE_DIR = {
    "heavytrucktrailer": "truck",
    "heavytruckhead": "truck",
    "truck": "truck",
    "pedestrian": "pedestrian",
    "pedestri": "pedestrian",
    "pedestria": "pedestrian",
    "tricycle": "pedestrian",
    "car": "car",
    "forklift": "car",
    "gocart": "car",
    "otherbigvehicle": "car",
    "scrapper": "car",
    "stacker": "car",
    "gantry": "block",
    "hatchway": "block",
    "lockbox": "block",
    "peggingblock": "block",
    "static": "block",
    "coneblock": "block",
    "conebloc": "block",
    "cylindarblock": "block",
    "crane": "block",
    "flag": "block",
    "container": "block",
    "fans": "block",


}


BEV_FACTOR = 2
Y_MIN = -50#-30
Y_MAX = 50#30
X_MIN = -50#-10
X_MAX = 100#90
VOXEL_X_SIZE = 0.1
VOXEL_Y_SIZE = 0.1
RADIUS_1 = int(40 / VOXEL_X_SIZE * BEV_FACTOR)
RADIUS_2 = int(70 / VOXEL_X_SIZE * BEV_FACTOR)
IMG_WIDTH = int((X_MAX - X_MIN) / VOXEL_X_SIZE)
IMG_HEIGHT = int((Y_MAX - Y_MIN) / VOXEL_Y_SIZE)
CENTER_X = int((-X_MIN) / VOXEL_X_SIZE) * BEV_FACTOR
CENTER_Y = int((-Y_MIN) / VOXEL_Y_SIZE) * BEV_FACTOR


def angle_to_rotMat_2d(angle):
    rotMat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotMat
def center_to_corner_box2d(box):
    """
    Calculate the box corners in 2D
    """
    translation = box[0:2]
    size = box[2:4]
    w, l = size[0], size[1]
    trackletBox = np.array(
        [
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        ]
    )
    yaw = box[-1]
    rotMat = angle_to_rotMat_2d(yaw)
    cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
    box2d = cornerPosInVelo.transpose()
    return box2d[0:4]

def load_obstacle_labels(label_path):
    lidar_frame_label = LidarFrameLabel()
    with open(label_path, "r") as fr:
        text_format.Parse(fr.read(), lidar_frame_label)
    obstacles = lidar_frame_label.obstacles[:]
    return obstacles


def find_closest_detection_res(res_dir, ts):
    """
    find detection result under directory based on timestamp, with bisect
    :param:  res_dir, detection directory
    :param:  ts, timestamp in second
    """
    files = sorted(os.listdir(res_dir))
    index_l = 0
    index_r = len(files) - 1
    if len(files) == 0:
        print("{} has {} files".format(res_dir, len(files)), fg_col="red")
        return ""
    while index_l != index_r:
        index_m = int((index_l + index_r) / 2)
        filename = files[index_m]
        file_ts = float(filename.replace(".pb.txt", "")) / 1e9
        ts_diff = file_ts - float(ts)
        if abs(ts_diff) <= 0.0001:#0.05:
            return osp.join(res_dir, filename)
        else:
            if ts_diff > 0:
                index_r = index_m
            else:
                index_l = index_m
        if (index_r - index_l) == 1:
            filename = files[index_r]
            file_ts = float(filename.replace(".pb.txt", "")) / 1e9
            ts_diff = file_ts - float(ts)
            if abs(ts_diff) <= 0.0001:#0.05:
                return osp.join(res_dir, filename)
            else:
                return ""
    return ""

def check_in_polygon(vertices, px, py):
    """
    :param vertices: [[], [], [], []]
    :return:
    """
    is_in = False
    for i, corner in enumerate(vertices):
        next_i = i + 1 if i + 1 < len(vertices) else 0
        x1, y1 = corner
        x2, y2 = vertices[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in

def generate_gt(label_path, gt_path, save_path):
    closest_dict = {}
    timestamp = os.listdir(gt_path)#['1619923269.078985.txt']
    timestamp = [ts.replace('.txt', '') for ts in timestamp]
    cnt = 0
    for ts in timestamp:
        closest_fid_path = find_closest_detection_res(label_path, ts)
        if closest_fid_path == "":
            cnt += 1
            #print("{} has no labels".format(ts))
            continue
        closest_dict[ts] = closest_fid_path
        #fid = label_path2frame_id[closest_fid_path]
        obstacles = load_obstacle_labels(closest_fid_path)
        #print((len(obstacles),closest_fid_path))
        important_box = []
        for obj in obstacles:
            if obj.type_str.lower() not in BIG_OBJECT_TYPES and obj.type_str.lower() not in SMALL_OBJECT_TYPES:
                continue
            vertices2d = center_to_corner_box2d(np.array([obj.x, obj.y, obj.width, obj.length, obj.rotation]))
            important_box.append([vertices2d, obj.type_str])
        labeled_grid = []
        #print(len(important_box))
        for box, obj_type in important_box:
            lt, lb, rb, rt = box
            min_x = np.min([lt[0], lb[0], rb[0], rt[0]])
            max_x = np.max([lt[0], lb[0], rb[0], rt[0]])
            min_y = np.min([lt[1], lb[1], rb[1], rt[1]])
            max_y = np.max([lt[1], lb[1], rb[1], rt[1]])
            if max_x < X_MIN or min_x >= X_MAX or max_y < Y_MIN or min_y >= Y_MAX:
                continue
            min_x = max(min_x, X_MIN)
            max_x = min(max_x, X_MAX - 1)
            min_y = max(min_y, Y_MIN)
            max_y = min(max_y, Y_MAX - 1)
            grid_min_x = int((min_x - X_MIN) / GRID_X)
            grid_max_x = int((max_x - X_MIN) / GRID_X)
            grid_min_y = int((min_y - Y_MIN) / GRID_Y)
            grid_max_y = int((max_y - Y_MIN) / GRID_Y)
            #print(obj_type,box)
            #print((grid_min_x, grid_max_x))
            #print((min_x, max_x))
            #print((grid_min_y, grid_max_y))
            #print((min_y, max_y))
            for i in range(grid_min_x, grid_max_x+1):
                for j in range(grid_min_y, grid_max_y+1):
                    centerpointx = x[j][i] + GRID_X/2
                    centerpointy = y[j][i] + GRID_Y/2
                    #print('**',centerpointx, centerpointy)
                    if check_in_polygon(box, centerpointx, centerpointy):
                        labeled_grid.append([i, j, obj_type])
        labeled_grid = np.array(labeled_grid)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, ts+'.txt')
        #print(labeled_grid.shape)
        np.savetxt(save_name, labeled_grid, fmt="%s")
    print("matched timestamp cnt:{}/{}".format(len(timestamp) - cnt, len(timestamp)))

    return closest_dict


@njit(cache=True, fastmath=True)
def gt_mode_jit(gridmap, x_range, y_range):
    grids0503 = []
    for i in range(0, x_range - 1, 5):
        for j in range(0, y_range - 1, 3):
            #cate_cnt = 0
            # print(gridmap.shape)
            cate_cnt = np.sum(gridmap[i:i+5, j:j+3])
            #for ii in range(i, i + 5):
            #    for jj in range(j, j + 3):
            #        cate_cnt += gridmap[ii][jj]
            if cate_cnt > 0:  # 3*5/2:
                grids0503.append((i // 5, j // 3))
    return grids0503


def grid0101_to_0503(data_path, mode="gt"):
    x_range, y_range = 1500, 1000
    gridmap = np.zeros((x_range, y_range), dtype=np.int32)
    #print(gridmap.shape)

    #grids0503 = []
    if mode == "gt":
        gt_data = np.loadtxt(data_path, dtype=np.int32)  # [:,:2]
        gt_grids = [tuple(gg[:2]) for gg in gt_data if gg[2] == 1 and gg[4] == 0]
        for grid in gt_grids:
            gridmap[grid[0]][grid[1]] = 1
        return gt_mode_jit(gridmap, x_range, y_range)
    else:
        raise NotImplementedError

@njit(cache=True, fastmath=True)
def cal_grid_pcd(points):
    grid_pcd_cnt = np.zeros((int((X_MAX - X_MIN) / GRID_X), int((Y_MAX - Y_MIN) / GRID_Y)), dtype=np.int32)
    length = len(points)
    for i in range(length):
        px = points[i][0]
        py = points[i][1]
        pz = points[i][2]
        if X_MIN < px < X_MAX:
            if Y_MIN < py < Y_MAX:
                if pz > 0.2:
                    gx = int((px - X_MIN) / GRID_X)
                    gy = int((py - Y_MIN) / GRID_Y)
                    grid_pcd_cnt[gx][gy] += 1
    return grid_pcd_cnt

def generate_gt_with_pcd(label_path, gt_path, save_path, pcd_path, pcd_save_path):


    closest_dict = {}
    timestamp = os.listdir(gt_path)#['1619923269.078985.txt']
    timestamp = [ts.replace('.txt', '') for ts in timestamp]
    cnt = 0
    for ts in timestamp:
        closest_fid_path = find_closest_detection_res(label_path, ts)
        if closest_fid_path == "":
            cnt += 1
            #print("{} has no labels".format(ts))
            continue
        closest_dict[ts] = closest_fid_path
        #print(pcd_path, closest_fid_path)


        pcd_name = os.path.join(pcd_path, closest_fid_path.split('/')[-1].replace('.pb.txt','.bin'))
        #print(pcd_path, closest_fid_path)

        points = read_pcd(pcd_name)
        grid_pcd_cnt = cal_grid_pcd(points)

        

        cmd = "cp {} {}".format(pcd_name, os.path.join(pcd_save_path,ts+'.bin'))
        #print(cmd)
        #os.system(cmd)



        #fid = label_path2frame_id[closest_fid_path]
        obstacles = load_obstacle_labels(closest_fid_path)
        #print((len(obstacles),closest_fid_path))
        important_box = []
        for obj in obstacles:
            if obj.type_str.lower() not in CATE_DIR.keys():
                continue
            vertices2d = center_to_corner_box2d(np.array([obj.x, obj.y, obj.width, obj.length, obj.rotation]))
            important_box.append([vertices2d, CATE_DIR[obj.type_str.lower()]])
        labeled_grid = []
        temp = []
        #print(len(important_box))
        for box, obj_type in important_box:
            lt, lb, rb, rt = box
            min_x = np.min([lt[0], lb[0], rb[0], rt[0]])
            max_x = np.max([lt[0], lb[0], rb[0], rt[0]])
            min_y = np.min([lt[1], lb[1], rb[1], rt[1]])
            max_y = np.max([lt[1], lb[1], rb[1], rt[1]])
            if max_x < X_MIN or min_x >= X_MAX or max_y < Y_MIN or min_y >= Y_MAX:
                continue
            min_x = max(min_x, X_MIN)
            max_x = min(max_x, X_MAX - 1)
            min_y = max(min_y, Y_MIN)
            max_y = min(max_y, Y_MAX - 1)
            grid_min_x = int((min_x - X_MIN) / GRID_X)
            grid_max_x = int((max_x - X_MIN) / GRID_X)
            grid_min_y = int((min_y - Y_MIN) / GRID_Y)
            grid_max_y = int((max_y - Y_MIN) / GRID_Y)
            #print(obj_type,box)
            #print((grid_min_x, grid_max_x))
            #print((min_x, max_x))
            #print((grid_min_y, grid_max_y))
            #print((min_y, max_y))
        #    temp.append([grid_min_x, grid_max_x, grid_min_y, grid_max_y, obj_type])
        #temp = np.array(temp)
        #save_root = "/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev_10_classes0101/labeled_bbox_0503_extra"
        #save_name = os.path.join(save_root, ts+'.txt')
        #np.savetxt(save_name, temp, fmt="%s")


            for i in range(grid_min_x, grid_max_x+1):
                for j in range(grid_min_y, grid_max_y+1):
                    centerpointx = x[j][i] + GRID_X/2
                    centerpointy = y[j][i] + GRID_Y/2
                    #print('**',centerpointx, centerpointy)
                    if check_in_polygon(box, centerpointx, centerpointy):
                        if grid_pcd_cnt[i][j] > 0:
                            labeled_grid.append((i, j, obj_type))
                        else:
                            labeled_grid.append((i, j, obj_type+"_bg"))


        #generate ray_gt
        gt_name = os.path.join(gt_path, ts+'.txt')
        gt_grids = grid0101_to_0503(gt_name, mode="gt")
        obstacle_gt_grids = [a[:2] for a in labeled_grid]
        ray_grids = list(set(gt_grids) - set(obstacle_gt_grids))
        for rg in ray_grids:
            labeled_grid.append([rg[0], rg[1], 'ray_gt'])

        labeled_grid = np.array(labeled_grid)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, ts+'.txt')
        #print(labeled_grid.shape)
        np.savetxt(save_name, labeled_grid, fmt="%s")
    print("matched timestamp cnt:{}/{}".format(len(timestamp) - cnt, len(timestamp)))

    return closest_dict




def read_pcd(file_path):
    if file_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    elif file_path.endswith('.bin'):
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

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
def draw_bg(pcd_path):
    """
    background for bev includes point clouds, hdmap mask, reference distance,
    and color scheme for boxes
    """
    canvas = np.zeros(
        (IMG_HEIGHT * BEV_FACTOR, IMG_WIDTH * BEV_FACTOR, 3), dtype=np.float32
    )
    points = read_pcd(pcd_path)#np.loadtxt(pcd_path)
    bev = draw_point_cloud_jit(points, canvas)
    #bev = draw_hdmap_mask(bev, mask_path)


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


def vis_single_result(pcd_path, gt_path, save_path, obstacle_gt_path, label_path):
    plt.figure("pcd and grid gt",dpi=300,figsize=(20,24))

    bev1 = draw_bg(pcd_path)
    gt_data = np.loadtxt(gt_path, dtype=np.int32)#[:,:2]
    gt_grids = gt_data[:,:2]
    is_obstacle = gt_data[:, 2]
    is_valid = gt_data[:, 4]

    for i, grid in enumerate(gt_grids):
        #idx = int(grid[0])
        #idy = int(grid[1])
        idx = int(grid[0] // 5)
        idy = int(grid[1] // 3)
        # cnt_z = str(int(grid[2]))
        if is_obstacle[i] == 0 or is_valid[i] == 1:
            continue

        img1 = draw_rotate_rect(
            bev1,
            rect=[x[idy][idx] + GRID_X/2, y[idy][idx] + GRID_Y/2, GRID_X, GRID_Y, 0],
            texts=[],
            color=(239, 255, 0),
            line_width=1,
            font_size=0.3
        )

    plt.subplot(2, 1, 1), plt.title('gt')
    plt.imshow(img1.astype(np.uint8))
    plt.axis('off')


    if obstacle_gt_path is not None:
        obs_gt_data = np.loadtxt(obstacle_gt_path, dtype=str)  # [:,:2]
        if len(obs_gt_data.shape) == 1:
            obs_gt_data = np.array([obs_gt_data])
        obs_gt_grids = obs_gt_data[:, :2]
        obs_gt_label = obs_gt_data[:, 2]
        grids = np.array(obs_gt_grids)
        bev2 = draw_bg(pcd_path)
        for indx,grid in enumerate(grids):
            idx = int(grid[0])
            idy = int(grid[1])
            # cnt_z = str(int(grid[2]))
            label = obs_gt_label[indx]
            if label == "ray_gt":
                img2 = draw_rotate_rect(
                    bev2,
                    rect=[x[idy][idx] + GRID_X / 2, y[idy][idx] + GRID_Y / 2, GRID_X, GRID_Y, 0],
                    texts=[],
                    color=(255, 255, 0),
                    line_width=1,
                    font_size=0.6
                )
            elif 'bg' in label:
                img2 = draw_rotate_rect(
                    bev2,
                    rect=[x[idy][idx] + GRID_X / 2, y[idy][idx] + GRID_Y / 2, GRID_X, GRID_Y, 0],
                    texts=[],
                    color=(255, 0, 0),
                    line_width=1,
                    font_size=0.6
                )
            else:

                img2 = draw_rotate_rect(
                    bev2,
                    rect=[x[idy][idx] + GRID_X / 2, y[idy][idx] + GRID_Y / 2, GRID_X, GRID_Y, 0],
                    texts=[],
                    color=(0, 255, 0),
                    line_width=1,
                    font_size=0.6
                )


        if label_path is not None:
            obstacles = load_obstacle_labels(label_path)
            for obj in obstacles:
                img2 = draw_rotate_rect(
                    bev2,
                    rect=[obj.x, obj.y,  obj.length, obj.width, obj.rotation],
                    texts=[obj.type_str],
                    color=(255, 255, 0),
                    line_width=1,
                    font_size=0.6
                )
        plt.subplot(2, 1, 2), plt.title('obstacle gt')
        plt.imshow(img2.astype(np.uint8))
        plt.axis('off')


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_name = os.path.join(save_path, gt_path.split('/')[-1].replace('.txt', '.jpg'))
    #cv2.imwrite(save_name, img2)
    plt.savefig(save_name)
    plt.clf()
    #plt.show()




GRID_X = 0.5#0.1
GRID_Y = 0.3#0.1
X = np.arange(X_MIN, X_MAX, GRID_X)#(-10, 91, 0.5)
shape_X = X.shape[0] - 1

Y = np.arange(Y_MIN, Y_MAX, GRID_Y)#(-30, 31, 0.3)
shape_Y = Y.shape[0] - 1
x, y = np.meshgrid(X, Y)
#print(X.shape, Y.shape, x.shape, y.shape)

factor_x = 1 / VOXEL_X_SIZE * BEV_FACTOR
factor_y = 1 / VOXEL_Y_SIZE * BEV_FACTOR
data_path = '/private/personal/linyuqi/grid_benchmark'
bag_name_list0 = sorted(os.listdir(data_path))#[:97]#[-100:]#['howo12_2021_05_29_15_04_30_40']#[sorted(os.listdir(data_path))[-1]]
#bag_name_list = [bag_name_list[9], bag_name_list[65]]
bag_name_list = bag_name_list0[56:]#[]
#for b in bag_name_list0:
#    if b.startswith("howo12"):# or b.startswith("howo16"):
#        #continue
#        bag_name_list.append(b)

'''
pcd_root = '/private/personal/linyuqi/grid_benchmark/howo10_2021_05_02_10_41_02_114/pcd/'
gt_root = '/private/personal/linyuqi/grid_benchmark/howo10_2021_05_02_10_41_02_114/gt/'
vis_save_path = '/private/personal/linyuqi/grid_benchmark/howo10_2021_05_02_10_41_02_114/vis_gt/'
obstacle_gt_save_path = '/private/personal/linyuqi/grid_benchmark/howo10_2021_05_02_10_41_02_114/obstacle_gt/'
msg_path = "/onboard_data/bags/meishangang/howo10/20210502/0846/howo10_2021_05_02_10_41_02_114.msg"
label_path = msg_path.replace('bags', 'point_cloud')
label_path = label_path.replace('.msg', '/LIDAR_OBSTACLES/')
'''
#cmp_root = '/private/personal/linyuqi/deep_lidar_grid_map/vis_result_rain_gt/all_7k'
#cmp_list0 = os.listdir(cmp_root)
#cmp_list = [xx.replace('.jpg','.txt') for xx in cmp_list0]
#for xx in cmp_list:
#    file_list.remove(xx)
#print(len(file_list))
#print(file_list[0])
#pkl_infos = pickle.load(open('howo_val_20210804.pkl', "rb"))
#fids = sorted(list(pkl_infos.keys()))
pkl_path = '/data/linyuqi/work/howo_val_20210804.pkl'
"""
def generate_msg_path_from_pkl(pkl_path):
    pkl_infos = pickle.load(open(pkl_path, "rb"))
    fids = sorted(list(pkl_infos.keys()))
    msg_paths = []
    for fid in fids:
        msg_paths.append(generate_msg_path(pkl_infos[fid]['MergePointCloudEgo']))
    return msg_paths


def generate_msg_path(merge_point_cloud_ego_path):
    paths = merge_point_cloud_ego_path.split('/')
    assert len(paths) == 10, 'this path not suitable to generate msg path'
    #print(len(paths),paths[0])
    msg_path = '/' + os.path.join(paths[1], 'bags', paths[3], paths[4], paths[5], paths[6], paths[7]+'.msg')
    #print(msg_path)
    return msg_path


BAG_PATHS = generate_msg_path_from_pkl(pkl_path)
BAG_PATHS = sorted(list(set(BAG_PATHS)))#[:60]
bag_name_to_path_dict = {}
for bp in BAG_PATHS:
    bn = bp.split('/')[-1].replace('.msg', '')
    bag_name_to_path_dict[bn] = bp
"""
def generate_path_from_pkl(pkl_path):
    pkl_infos = pickle.load(open(pkl_path, "rb"))
    fids = sorted(list(pkl_infos.keys()))
    bag_name_to_pcd_path = {}
    bag_name_to_label_path = {}
    for fid in fids:
        bn = pkl_infos[fid]['MergePointCloudEgo'].split('/')[7]
        pcd_path = os.path.dirname(pkl_infos[fid]['MergePointCloudEgo'])
        label_path = os.path.dirname(pkl_infos[fid]['LidarObstaclesLabeled'])
        if bn not in bag_name_to_pcd_path.keys():
            bag_name_to_pcd_path[bn] = pcd_path
            bag_name_to_label_path[bn] = label_path
    return bag_name_to_pcd_path, bag_name_to_label_path

bag_name_to_pcd_path, bag_name_to_label_path = generate_path_from_pkl(pkl_path)



print("{} bags to process".format(len(bag_name_list)))
for idx,bag_name in enumerate(bag_name_list):#(bag_name_to_path_dict.keys()):#(bag_name_list):
    print("To process {}th bag:{}".format(idx, bag_name))
    pcd_root = bag_name_to_pcd_path[bag_name]#os.path.join(data_path, bag_name, 'pcd')
    label_path = bag_name_to_label_path[bag_name]

    gt_root = os.path.join(data_path, bag_name, 'gt')
    obstacle_gt_save_path = os.path.join(data_path, bag_name, 'obstacle_gt_fine0503')
    pcd_save_path = os.path.join(data_path, bag_name, 'pcd_fine')
    #pcd_save_path = "/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev_10_classes0101/images_fine/extra"

    '''
    if os.path.exists(obstacle_gt_save_path):
        cmd1 = "rm {}/*".format(obstacle_gt_save_path)
        os.system(cmd1)
    if os.path.exists(pcd_save_path):
        cmd2 = "rm {}/*".format(pcd_save_path)
        os.system(cmd2)
    if not os.path.exists(pcd_save_path):
        os.makedirs(pcd_save_path)
    '''

    #msg_path = bag_name_to_path_dict[bag_name]
    #label_path = msg_path.replace('bags', 'point_cloud')
    #label_path = label_path.replace('.msg', '/LIDAR_OBSTACLES/')


    closest_dict = generate_gt_with_pcd(label_path, gt_root, obstacle_gt_save_path, pcd_root, pcd_save_path)
    #closest_dict = generate_gt(label_path, gt_root, obstacle_gt_save_path)

    if idx < 1000:
        file_list = os.listdir(obstacle_gt_save_path)#['1619946404.285842.txt']#os.listdir(res_root)
        file_list = sorted([os.path.splitext(temp)[0] for temp in file_list])
        #print(len(file_list))


        vis_save_path = os.path.join(data_path, bag_name, 'vis_gt_fine0503')
        if not os.path.exists(vis_save_path):
            os.makedirs(vis_save_path)
        show_list = file_list[::10]
        for file in tqdm(show_list):
            pcd_path = os.path.join(pcd_save_path, file+'.bin')#os.path.join(pcd_root, file+'.pcd')
            gt_path = os.path.join(gt_root, file+'.txt')
            obstacle_gt_path = os.path.join(obstacle_gt_save_path, file+'.txt')
            fid_path = closest_dict[file]

            if os.path.exists(pcd_path):
                vis_single_result(pcd_path, gt_path, vis_save_path, obstacle_gt_path, fid_path)




