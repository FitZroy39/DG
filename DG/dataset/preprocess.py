import numpy as np
from configs import IMG_SIZE, AREA_EXTENTS, voxel_size
import torch
import cv2

map_y, map_x = IMG_SIZE
cx = round((0 - AREA_EXTENTS[0][0]) / voxel_size[0])
cy = round((0 - AREA_EXTENTS[1][0]) / voxel_size[1])
rot_center = (cx, cy)

def jitter_point(point, sigma=0.05, clip=0.05):
    assert(clip > 0)
    #point = np.array(point)
    #point = point.reshape(-1,3)
    Row, Col = point.shape
    jittered_point = np.clip(sigma * np.random.randn(Row, Col), -1*clip, clip)
    #jittered_point += point
    point += jittered_point
    return point#jittered_point


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False
def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def global_rotation_voxel_feat(voxel_feat, theta, scale=1.0):
    ny, nx, c = voxel_feat.shape  # H W C
    # rot_mat = get_rotation_scale_matrix2d((int(nx / 2), int(ny / 2)), theta, scale)
    rot_mat = cv2.getRotationMatrix2D(rot_center, -theta * 180 / np.pi, scale)
    voxel_feat = cv2.warpAffine(voxel_feat, rot_mat, (nx, ny), flags=cv2.INTER_NEAREST)  # H W C
    return voxel_feat.reshape(ny, nx, -1)

def global_scale_voxel_feat(voxel_feat, scale, theta=0.0):
    ny, nx, c = voxel_feat.shape  # H W C
    # rot_mat = get_rotation_scale_matrix2d((int(nx / 2), int(ny / 2)), theta, scale)
    rot_mat = cv2.getRotationMatrix2D(rot_center, -theta * 180 / np.pi, scale)
    voxel_feat = cv2.warpAffine(voxel_feat, rot_mat, (nx, ny), flags=cv2.INTER_NEAREST)  # H W C
    return voxel_feat.reshape(ny, nx, -1)

def random_flip_along_x(gt_seg, points, observations):
    """
    Args:
        gt_seg: 500, 1000, 1
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        points[:, 1] = -points[:, 1]
        gt_seg = np.flipud(gt_seg)
        if observations is not None:
            observations = np.flipud(observations)
    return gt_seg, points, observations

def global_rotation(gt_seg, points, observations, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_seg = global_rotation_voxel_feat(gt_seg, noise_rotation)
    if observations is not None:
        observations = global_rotation_voxel_feat(observations, noise_rotation)
    return gt_seg, points, observations


def global_scaling(gt_seg, points, observatjions, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_seg, points, observatjions
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_seg = global_scale_voxel_feat(gt_seg, noise_scale)
    if observatjions is not None:
        observatjions = global_scale_voxel_feat(observatjions, noise_scale)
    return gt_seg, points, observatjions


def global_translate_voxel_feat(voxel_feat, dw, dh):
    ny, nx, c = voxel_feat.shape  # H W C
    mat = np.array([[1, 0, dw], [0, 1, dh]], dtype=np.float32)
    voxel_feat = cv2.warpAffine(voxel_feat, mat, (nx, ny), flags=cv2.INTER_NEAREST)  # H W C
    return voxel_feat.reshape(ny, nx, -1)


def global_translate(gt_seg, points, observations, noise_translate_std):
    """
    Apply global translation to gt_boxes and points.
    """
    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array([noise_translate_std, noise_translate_std, noise_translate_std])

    std_x, std_y, std_z = noise_translate_std
    noise_translate = np.array([np.random.normal(0, std_x, 1),
                                np.random.normal(0, std_y, 1),
                                np.random.normal(0, std_z, 1)]).T  # 1 3
    # clip to 3*std
    noise_translate = np.clip(noise_translate, [-3.0*std_x, -3.0*std_y, -3.0*std_z], [3.0*std_x, 3.0*std_y, 3.0*std_z])
    points[:, :3] += noise_translate

    dw = noise_translate[0, 0] // 0.1
    dh = noise_translate[0, 1] // 0.1
    gt_seg = global_translate_voxel_feat(gt_seg, dw, dh)
    if observations is not None:
        observations = global_translate_voxel_feat(observations, dw, dh)

    return gt_seg, points, observations
"""
def rotation_points_single_angle(points, angle, axis=0):
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype)
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype)
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype)
    else:
        raise ValueError("axis should in range")

    return points @ rot_mat_T


def random_flip(gt_gridmap, points, probability=0.5):
    assert gt_gridmap.shape == IMG_SIZE
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability])
    if enable:
        gt_gridmap = np.flip(gt_gridmap, 0)

        points[:, 1] = -points[:, 1]
    return gt_gridmap, points


def global_scaling_v2(gt_gridmap, points, min_scale=0.95, max_scale=1.05):
    assert gt_gridmap.shape == IMG_SIZE
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    #gt_boxes[:, :6] *= noise_scale
    return gt_gridmap, points


def global_rotation_v2(gt_gridmap, points, min_rad=-np.pi / 4,
                       max_rad=np.pi / 4):
    assert gt_gridmap.shape == IMG_SIZE
    noise_rotation = np.random.uniform(min_rad, max_rad)
    points[:, :3] = rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2)
    #gt_boxes[:, :3] = rotation_points_single_angle(
    #    gt_boxes[:, :3], noise_rotation, axis=2)
    #gt_boxes[:, 6] += noise_rotation
    return gt_gridmap, points

def global_translate(gt_gridmap, points, noise_translate_std):
    '''
    Apply global translation to gt_boxes and points.
    '''
    assert gt_gridmap.shape == IMG_SIZE

    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array([noise_translate_std, noise_translate_std, noise_translate_std])

    noise_translate = np.array([np.random.normal(0, noise_translate_std[0], 1),
                                np.random.normal(0, noise_translate_std[1], 1),
                                np.random.normal(0, noise_translate_std[0], 1)]).T

    points[:, :3] += noise_translate
    #gt_boxes[:, :3] += noise_translate

    return gt_gridmap, points
"""