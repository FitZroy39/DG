import os
from tqdm import tqdm

dirty_frames = [
    '1622647736.599723.txt',
    '1622647736.901554.txt',
    '1622647737.004266.txt',
    '1622647737.300124.txt',
    '1622647737.402273.txt',
    '1622647737.499497.txt',
    '1622647737.499497.txt',
    '1622647738.900925.txt',
    '1622647738.997864.txt',
    '1622647739.101820.txt',
    '1622647739.197722.txt',
    '1622647739.299260.txt',
    '1622647739.402592.txt',
    '1622647739.501463.txt',
    '1622647739.608993.txt',
    '1622647739.712597.txt',
    '1622647740.002018.txt',
    '1622647740.101010.txt',
    '1622647742.703187.txt',
    '1622647742.999854.txt',
    '1622647743.499901.txt',
    '1622647743.900822.txt',
    '1622647744.102026.txt',
]

def is_dirty(x):
    if '1622647290.100976.txt' <= x <= '1622647297.200068.txt' or \
    '1622647324.301211.txt' <= x <= '1622647327.199202.txt' or \
    '1622647231.701940.txt' <= x <= '1622647232.601849.txt' or \
    '1622645724.205283.txt' <= x <= '1622645727.303753.txt' or \
    '1622645659.401288.txt' <= x <= '1622645664.102010.txt' or \
    '1622645326.004329.txt' <= x <= '1622645330.400525.txt' or \
    '1622645191.504692.txt' <= x <= '1622645200.309240.txt' or \
    '1622641897.006290.txt' <= x <= '1622641908.904747.txt' or \
    '1622639780.614338.txt' <= x <= '1622639784.309719.txt':
        return True

    else:
        return False





ann_root = '/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/linyuqi/data/pc_bev_2classes0503_300x300/annotations/training'
pcd_root = '/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/linyuqi/data/pc_bev_2classes0503_300x300/images/training'
pred_root = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/linyuqi/code/deep_lidar_model/preds/grid_benchmark0503_test_9w_cw_190x200_final3"

bag_root = '/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/linyuqi/grid_benchmark0503_300x300/grid_benchmark0503_300x300'
no_fog_bag_list = ['howo13_2021_06_02_21_54_38_43',
'howo13_2021_06_02_22_18_38_67',
'howo13_2021_06_02_22_19_38_68',
'howo13_2021_06_02_23_29_38_138']

bag_list = os.listdir(bag_root)
#fog_frames_list = []

train_ann = os.listdir(pcd_root)
print('train ann before upsample:{}'.format(len(train_ann)))
for bag in tqdm(bag_list):
    if bag.startswith('howo13_2021_06_02') and bag not in no_fog_bag_list:
        gt_path = os.path.join(pred_root, bag)
        pcd_path = os.path.join(bag_root, bag, 'pcd')
        gt_list = os.listdir(gt_path)
        #fog_frames_list.extend(gt_list)
        for gt in gt_list:
            if gt in dirty_frames or is_dirty(gt):
                continue
            gt_file = os.path.join(gt_path, gt)
            target_gt_path = os.path.join(ann_root, 'extra_'+gt)
            cmd1 = "cp {} {}".format(gt_file, target_gt_path)
            pcd_file = os.path.join(pcd_path, gt.replace('.txt','.pcd'))
            target_pcd_path = os.path.join(pcd_root, 'extra_'+gt.replace('.txt','.pcd'))
            cmd2 = "cp {} {}".format(pcd_file, target_pcd_path)
            #os.system(cmd1)
            #if os.path.exists(pcd_file):
            #    print(cmd2)
            #else:
            #    print('error')
            os.system(cmd2)
train_ann = os.listdir(pcd_root)
print('train ann after upsample:{}'.format(len(train_ann)))

'''
train_ann = os.listdir(ann_path)
print('train ann before upsample:{}'.format(len(train_ann)))

for ann in tqdm(train_ann):
    if ann in fog_frames_list:
        source_ann = os.path.join(ann_path, ann)
        target_ann = os.path.join(ann_path, 'extra2_'+ann)
        cmd1 = "cp {} {}".format(source_ann, target_ann)

        source_pcd = os.path.join(pcd_path, ann.replace('.txt','.pcd'))
        target_pcd = os.path.join(pcd_path, 'extra2_'+ann.replace('.txt', '.pcd'))
        cmd2 = "cp {} {}".format(source_pcd, target_pcd)

        os.system(cmd1)
        os.system(cmd2)

train_ann2 = os.listdir(ann_path)
print('train ann after upsample:{}'.format(len(train_ann2)))
'''