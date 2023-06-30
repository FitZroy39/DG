import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
from pathlib import Path
from model.hrnet.hrnet import get_hrnet
from dataset.transform import PCBEVTransform
from pytorch_lightning.core.lightning import LightningModule
from dataset.pc_bev import ExtractBevFeature
from torchvision.transforms import functional as F
import open3d as o3d
from configs import SELF_CAR_EXTENTS,AREA_EXTENTS,voxel_size,log_norm,num_slice, DETECTION_AREA_EXTENTS, IMG_SIZE
from evaluation.obstacle_eval import get_obstacle_eval_result


def read_pcd(file_path):
	pcd = o3d.io.read_point_cloud(file_path)
	points = np.asarray(pcd.points)
	return points

class PCBEVModel(LightningModule):
    def __init__(self):
        super().__init__()
        cfg = './model/hrnet/conf_file/m1112_b2_c12.yaml'
        self.backbone = get_hrnet(cfg, upsample=True)
        self.transform = PCBEVTransform()

    def load_weights(self, model_path, strict=True):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'], strict=strict)

    def forward(self, x):
        x, _ = self.transform(x, None)
        features = self.backbone(x)
        return features

def img2bev(img_path, label_path):

    if img_path.endswith('.txt'):
        points = np.loadtxt(img_path)
    else:
        points = read_pcd(img_path)
    extract_bev = ExtractBevFeature(area_extents=AREA_EXTENTS,
                                    self_car_extents=SELF_CAR_EXTENTS,
                                    num_slices=num_slice,
                                    voxel_size=voxel_size,
                                    log_norm=log_norm)
    bev_img = extract_bev(points)
    img = bev_img


    if label_path.endswith('.pcd'):
        label_path = label_path.replace('.pcd', '.txt')#('images', 'annotations')
    gt_info = np.loadtxt(label_path, dtype=int)
    gt_semantic_seg = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.int32)
    length = len(gt_info)
    for i in range(length):
        x = int(gt_info[i][0])
        y = int(gt_info[i][1])
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

    return img, target

        
def eval_and_format_results(model_path, save_path, pcd_root, gt_root):
    save_root = save_path
    Path(save_root).mkdir(parents=True, exist_ok=True)

    file_list = os.listdir(pcd_root)#[:10]  # ['1619946404.285842.txt']#os.listdir(res_root)
    print("{} files to evaluate and format results".format(len(file_list)))

    model = PCBEVModel()
    model.load_weights(model_path)
    model = model.cuda()
    model.eval()

    for file in file_list:
        pcd_path = os.path.join(pcd_root, file)
        gt_path = os.path.join(gt_root, file)
        img, target = img2bev(pcd_path, gt_path, is_300x300=True)
        img = F.to_tensor(img).cuda()
        with torch.no_grad():
            prediction = model(img.unsqueeze(0))

        prediction = torch.argmax(prediction.cpu(), dim=1)[0]
        prediction = prediction.numpy()

        obs_grids = []
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                if prediction[i][j] == 1:
                    obs_grids.append([i+DETECTION_AREA_EXTENTS[0][0], j+DETECTION_AREA_EXTENTS[1][0], 1, 1, 0, 1, 1])
        obs_grids = np.array(obs_grids, dtype=np.int32)
        save_name = os.path.join(save_root, file)
        if save_name.endswith(".pcd"):
            save_name = save_name.replace(".pcd", ".txt")
        np.savetxt(save_name, obs_grids, fmt="%d")

if __name__ == '__main__':
    evaluate = False#True
    speed_test = False
    format_reults = True
    model_path = '/mnt/yrfs/yanrong/pvc-34488cf7-703b-4654-9fe8-762a747bbc58/wangtiantian/project/deep-lidar-model/logs2/deepgrid1202_2/epoch=06-Accu=0.89.ckpt'
    data_root = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/linyuqi/grid_benchmark0503_300x300/grid_benchmark0503_300x300_val"
    save_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/20w_170x270_eval"

    if format_reults:
        bag_list = sorted(os.listdir(data_root))
        for idx, bag in enumerate(bag_list):
            print("===>{}/{} bag:{}".format(idx, len(bag_list), bag))
            pcd_path = os.path.join(data_root, bag, "pcd")
            gt_path = os.path.join(data_root, bag, "gt")
            save_path = os.path.join(save_dir, bag)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            eval_and_format_results(model_path, save_path, pcd_path, gt_path)

    if evaluate:
        obstacle_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/deep-lidar-model/preds/20w_300x300_final2_obstacle"
        gt_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/eva/gt"
        get_obstacle_eval_result(obstacle_dir, save_dir, gt_dir)

    if speed_test:
        from utils.utils import model_speed_test
        
        model = PCBEVModel()
        model.load_weights(model_path)
        model = model.cuda()
        model_speed_test(model, 100)

