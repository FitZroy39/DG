import os
import numpy as np
from config import SELF_CAR_EXTENTS, AREA_EXTENTS, voxel_size
from metrics import intersect_and_union_range, pre_eval_to_metrics
from collections import OrderedDict
from prettytable import PrettyTable
from tqdm import tqdm
import sys
sys.path.append("..") 
from configs import IMG_SIZE, DETECTION_AREA_EXTENTS

INNER_RANGE_X = [-5, 30]
INNER_RANGE_Y = [-30, 30]

class Eval_Range:
    def __init__(self, pred_dir, gt_dir, class_names, ignore_index=255, label_map=dict(), reduce_zero_label=False):
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index
        self.label_map = label_map
        self.reduce_zero_label = reduce_zero_label

    def load_and_transform_txt(self, filename):
        data = np.loadtxt(filename, dtype=np.int32)

        semantic_seg = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 1), dtype=np.int32)
        length = len(data)

        is_obstacle = data[:, 2]
        is_valid = data[:, 4]
        for i in range(length):
            if is_obstacle[i] == 0 or is_valid[i] == 1:
                continue
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



    def pre_eval_range(self):
        baseline_dir = self.gt_dir.replace('/gt', '/baseline_gt')
        pred_file_list = os.listdir(baseline_dir)  # (self.pred_dir)#(self.gt_dir)
        print("{} files to evaluate".format(len(pred_file_list)))

        pre_eval_results_inner = []
        pre_eval_results_outer = []
        inner_range = np.array([INNER_RANGE_X, INNER_RANGE_Y])
        max_coord_x = int((AREA_EXTENTS[0][1] - AREA_EXTENTS[0][0]) // voxel_size[0]) #200
        max_coord_y = int((AREA_EXTENTS[1][1] - AREA_EXTENTS[1][0]) // voxel_size[1])  # 200
        mask = np.zeros((max_coord_x, max_coord_y))

        for i in range(len(inner_range)):
            inner_range[i] = (inner_range[i] - AREA_EXTENTS[i][0]) / voxel_size[i]

        mask[inner_range[0][0]:inner_range[0][1], inner_range[1][0]:inner_range[1][1]] = 1
        mask_inner = (mask == 1)
        mask_outer = (mask == 0)
        for file in tqdm(pred_file_list):
            pred_file = os.path.join(self.pred_dir, file)
            gt_file = os.path.join(self.gt_dir, file)
            assert os.path.isfile(gt_file), "gt file not exist"

            pred_seg_map = self.load_and_transform_txt(pred_file, is_300x300=True)
            gt_seg_map = self.load_and_transform_txt(gt_file, is_300x300=True)

            pre_eval_results_inner.append(
                intersect_and_union_range(pred_seg_map, gt_seg_map, self.num_classes,
                                          mask_inner, self.label_map,
                                          self.reduce_zero_label))
            pre_eval_results_outer.append(
                intersect_and_union_range(pred_seg_map, gt_seg_map, self.num_classes,
                                          mask_outer, self.label_map,
                                          self.reduce_zero_label))

        return pre_eval_results_inner, pre_eval_results_outer


    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = metric.split(',')  # [metric]

        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        obstacle_results = {}
        ret_metrics = pre_eval_to_metrics(results, metric)


        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': self.class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)
            obstacle_results[key] = val[-1]

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print('per class results:')
        print('\n' + class_table_data.get_string())
        print('Summary:')
        print('\n' + summary_table_data.get_string())

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(self.class_names)
            })

        return eval_results, obstacle_results




if __name__ == "__main__":
    pred_root = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/lr40"
    gt_root = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/linyuqi/grid_benchmark0503_300x300/grid_benchmark0503_300x300_val"
    
    class_names = ['background', 'obstacle']
    bag_list = sorted(os.listdir(pred_root))
    inner_iou = []
    inner_presicion = []
    inner_recall = []
    inner_f1 = []

    outer_iou = []
    outer_presicion = []
    outer_recall = []
    outer_f1 = []

    for bag in bag_list:
        print("==>evaluating bag:{}".format(bag))
        pred_dir = os.path.join(pred_root, bag,)# "baseline_gt")
        gt_dir = os.path.join(gt_root, bag, "gt")

        eval_range = Eval_Range(pred_dir, gt_dir, class_names,)
        pre_eval_results_inner, pre_eval_results_outer = eval_range.pre_eval_range()

        _,eval_results_inner = eval_range.evaluate(pre_eval_results_inner, metric=['mIoU', 'mDice', 'mFscore'])
        for k in eval_results_inner.keys():
            if k == "IoU":
                inner_iou.append(eval_results_inner[k])
            elif k == "Precision":
                inner_presicion.append(eval_results_inner[k])
            elif k == "Recall":
                inner_recall.append(eval_results_inner[k])
            elif k == "Fscore":
                inner_f1.append(eval_results_inner[k])

        _,eval_results_outer = eval_range.evaluate(pre_eval_results_outer, metric=['mIoU', 'mDice', 'mFscore'])
        for k in eval_results_outer.keys():
            if k == "IoU":
                outer_iou.append(eval_results_outer[k])
            elif k == "Precision":
                outer_presicion.append(eval_results_outer[k])
            elif k == "Recall":
                outer_recall.append(eval_results_outer[k])
            elif k == "Fscore":
                outer_f1.append(eval_results_outer[k])

    print('\n')
    print("****results inner range x:{},y:{}****".format(INNER_RANGE_X, INNER_RANGE_Y))
    print('\n')
    print("-------------iou of {} bags--------------".format(len(bag_list)))
    print(inner_iou)
    print("mean:", np.mean(inner_iou))
    print("-------------precision of {} bags--------------".format(len(bag_list)))
    print(inner_presicion)
    print("mean:", np.mean(inner_presicion))
    print("-------------recall of {} bags--------------".format(len(bag_list)))
    print(inner_recall)
    print("mean:", np.mean(inner_recall))
    print("-------------fscore of {} bags--------------".format(len(bag_list)))
    print(inner_f1)
    print("mean:", np.mean(inner_f1))

    print('\n')
    print("****results outer range x:{},y:{}****".format(INNER_RANGE_X, INNER_RANGE_Y))
    print('\n')
    print("-------------iou of {} bags--------------".format(len(bag_list)))
    print(outer_iou)
    print("mean:", np.mean(outer_iou))
    print("-------------precision of {} bags--------------".format(len(bag_list)))
    print(outer_presicion)
    print("mean:", np.mean(outer_presicion))
    print("-------------recall of {} bags--------------".format(len(bag_list)))
    print(outer_recall)
    print("mean:", np.mean(outer_recall))
    print("-------------fscore of {} bags--------------".format(len(bag_list)))
    print(outer_f1)
    print("mean:", np.mean(outer_f1))