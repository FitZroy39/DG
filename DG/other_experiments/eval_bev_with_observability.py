import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import numpy as np
from tqdm import tqdm
from glob import glob 
from pathlib import Path
from imageio import imsave, imread
from model.hrnet.hrnet import get_hrnet
from dataset.transform import PCBEVTransform
from pytorch_lightning.core.lightning import LightningModule
#from utils.visualize import sem_visulization, visualize_drivable_area
from utils.grid_vis_miss_and_false import vis_single_result
from dataset.pc_bev import ExtractBevFeature
from torchvision.transforms import functional as F
from utils.ransac import denoise
import open3d as o3d
from dataset.pc_bev import cal_grid_pcd, cal_observabilty

AREA_EXTENTS = [[-10, 90], [-30, 30], [-1, 2]]
voxel_size = (0.5, 0.3)
SELF_CAR_EXTENTS = [[-7.18, 0],[-1.85, 1.85]]
log_norm = 4
num_slice = 5


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

def img2bev(img_path):
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
    #img = bev_img#.transpose(2, 0, 1)

    grid_pcd_cnt = cal_grid_pcd(points)
    obs = cal_observabilty(points, grid_pcd_cnt)
    max_num = np.max(obs)
    # obs[obs > max_num] = max_num
    obs = obs / max_num
    img = np.concatenate((bev_img, obs), axis=-1)

    # target = {}
    label_path = img_path.replace('pcd/', 'gt/')
    if label_path.endswith('.pcd'):
        label_path = label_path.replace('.pcd', '.txt')#('images', 'annotations')
    gt_info = np.loadtxt(label_path, dtype=int)
    bev_x_h = round((AREA_EXTENTS[0][1] - AREA_EXTENTS[0][0]) / voxel_size[0])
    bev_y_w = round((AREA_EXTENTS[1][1] - AREA_EXTENTS[1][0]) / voxel_size[1])
    gt_semantic_seg = np.zeros((bev_x_h, bev_y_w, 1), dtype=np.int32)
    length = len(gt_info)
    for i in range(length):
        x = int(gt_info[i][0])
        y = int(gt_info[i][1])
        is_obstacle = int(gt_info[i][2])
        is_valid = int(gt_info[i][4])
        if is_obstacle == 1 and is_valid == 0:
            gt_semantic_seg[x, y, 0] = 1
            # gt_semantic_seg[x, y, 1] = 255
            # gt_semantic_seg[x, y, 2] = 255

    target = gt_semantic_seg.squeeze().astype(np.uint8)


    return img, target




def eval_and_vis(model_path, to_video=False):
    "will inference each frame in 'test_bags' and save visualizations to ./vis "

    pcd_root = '/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev_rain/images/validation/'
    gt_root = '/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev_rain/annotations/validation/'
    save_root = './vis/'
    Path(save_root).mkdir(parents=True, exist_ok=True)

    file_list = os.listdir(pcd_root)[:10]  # ['1619946404.285842.txt']#os.listdir(res_root)
    print("{} files to evaluate and vis".format(len(file_list)))

    model = PCBEVModel()
    model.load_weights(model_path)
    model = model.cuda()
    model.eval()

    for file in tqdm(file_list):
        save_path = os.path.join(save_root, file)
        pcd_path = os.path.join(pcd_root, file)
        gt_path = os.path.join(gt_root, file)

        img, target = img2bev(pcd_path)
        img = F.to_tensor(img).cuda()
        #img = torch.tensor(img).cuda()
        with torch.no_grad():
            prediction = model(img.unsqueeze(0))

        prediction = torch.argmax(prediction.cpu(), dim=1)[0]
        prediction = prediction.numpy()
        vis_single_result(pcd_path, prediction, gt_path, save_path)


    if to_video:
        cmd = "ffmpeg -framerate 30 -pattern_type glob -i './vis/*.jpg' -c:v libx264 -r 30 -y out.mp4"
        os.system(cmd)
        

def eval_and_format_results(model_path, save_path, pcd_root, gt_root):
    #pcd_root = '/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev/images/validation/'
    #gt_root = '/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev/annotations/validation/'
    save_root = save_path
    Path(save_root).mkdir(parents=True, exist_ok=True)

    file_list = os.listdir(pcd_root)#[:10]  # ['1619946404.285842.txt']#os.listdir(res_root)
    print("{} files to evaluate and format results".format(len(file_list)))

    model = PCBEVModel()
    model.load_weights(model_path)
    model = model.cuda()
    model.eval()

    for file in file_list:
        #save_path = os.path.join(save_root, file)
        pcd_path = os.path.join(pcd_root, file)
        #gt_path = os.path.join(gt_root, file)

        img, target = img2bev(pcd_path)
        img = F.to_tensor(img).cuda()
        # img = torch.tensor(img).cuda()
        with torch.no_grad():
            prediction = model(img.unsqueeze(0))

        prediction = torch.argmax(prediction.cpu(), dim=1)[0]
        prediction = prediction.numpy()

        obs_grids = []
        for i in range(prediction.shape[0]):
            for j in range(prediction.shape[1]):
                if prediction[i][j] == 1:
                    obs_grids.append([i, j, 1, 1, 1, 1, 1])
        obs_grids = np.array(obs_grids, dtype=np.int32)
        save_name = os.path.join(save_root, file)
        #print(save_name, save_name.endswith(".pcd"))
        if save_name.endswith(".pcd"):
            save_name = save_name.replace(".pcd", ".txt")
        np.savetxt(save_name, obs_grids, fmt="%d")

if __name__ == '__main__':
    evaluate = False#True
    speed_test = True#False
    format_reults = False#True
    model_path = '/private/personal/linyuqi/gpu12/deep_lidar_model/logs/test_run_binclass05x03_lovaszloss_cw_observability/epoch=10-Accu=0.87.ckpt'#'/private/personal/linyuqi/gpu12/port_visual_model/logs/test_run3_nodownsample/epoch=11-Accu=0.85.ckpt'

    if evaluate:
        eval_and_vis(model_path)

    if format_reults:
        data_root = "/private/personal/linyuqi/grid_benchmark0503_test"#"/private/personal/linyuqi/grid_benchmark0503"
        bag_list = sorted(os.listdir(data_root))#[:10]
        #save_path = model_path.replace('.ckpt', '_results')
        for idx, bag in enumerate(bag_list):
            print("===>{}/{} bag:{}".format(idx, len(bag_list), bag))
            pcd_path = os.path.join(data_root, bag, "pcd")
            gt_path = os.path.join(data_root, bag, "gt")
            save_path = os.path.join("/private/personal/linyuqi/gpu12/deep_lidar_model/preds/grid_benchmark0503_test_observability", bag)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            eval_and_format_results(model_path, save_path, pcd_path, gt_path)


    if speed_test:
        from utils.utils import model_speed_test
        
        model = PCBEVModel()
        model.load_weights(model_path)
        model = model.cuda()
        model_speed_test(model, 100, observability=True)

        #test_img = torch.randn(1, 3, 1920, 1200).cuda()
        #test_img, target = img2bev("/private/personal/linyuqi/grid_benchmark0503/howo10_2021_05_01_12_52_39_5/pcd/1619844760.919672.pcd")
        #test_img = F.to_tensor(test_img).cuda().unsqueeze(0)
        #model_speed_test(model, test_img, 100)
