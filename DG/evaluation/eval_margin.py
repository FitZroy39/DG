import os
import numpy as np
from metrics import intersect_and_union, pre_eval_to_metrics
from collections import OrderedDict
from prettytable import PrettyTable
from tqdm import tqdm
import sys
sys.path.append("..") 
from configs import IMG_SIZE, DETECTION_AREA_EXTENTS

class Custom_Eval:
    def __init__(self, pred_dir, gt_dir, class_names, ignore_index=255, label_map=dict(), reduce_zero_label=False):
        self.pred_dir = pred_dir
        self.gt_dir = gt_dir
        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.ignore_index = ignore_index
        self.label_map = label_map
        self.reduce_zero_label = reduce_zero_label

    def compare_grid(self, gt_file, pred_seg):
        data = np.loadtxt(gt_file, dtype=np.int32)
        compare_seg = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.int32)
        length = len(data)
        is_obstacle = data[:, 3]
        margin_left = data[:, 6]
        num_grid = 0
        for i in range(length):
            if is_obstacle[i] == 0:
                continue
            x = data[i][0]
            y = data[i][1]
            if x < DETECTION_AREA_EXTENTS[0][0] or x >= DETECTION_AREA_EXTENTS[0][1] or y < DETECTION_AREA_EXTENTS[1][0] or y >= DETECTION_AREA_EXTENTS[1][1]:
                continue
            else:
                x = x - DETECTION_AREA_EXTENTS[0][0]
                y = y - DETECTION_AREA_EXTENTS[1][0]
            if (pred_seg[x][y] == 0):
                continue
            num_grid += 1
            compare_seg[x, y] = abs(pred_seg[x][y] - margin_left[i])

        return compare_seg, num_grid

    def load_and_transform_txt(self, filename):
        data = np.loadtxt(filename, dtype=np.int32)

        semantic_seg = np.zeros((IMG_SIZE[0], IMG_SIZE[1]), dtype=np.int32)
        length = len(data)

        is_obstacle = data[:, 3]
        margin_left = data[:, 6]
        num_grid = 0
        # is_valid = data[:, 4]
        for i in range(length):
            # if is_obstacle[i] == 0 or is_valid[i] == 1:
            if is_obstacle[i] == 0:
                continue
            x = data[i][0]
            y = data[i][1]

            if x < DETECTION_AREA_EXTENTS[0][0] or x >= DETECTION_AREA_EXTENTS[0][1] or y < DETECTION_AREA_EXTENTS[1][0] or y >= DETECTION_AREA_EXTENTS[1][1]:
                continue
            else:
                x = x - DETECTION_AREA_EXTENTS[0][0]
                y = y - DETECTION_AREA_EXTENTS[1][0]
            num_grid += 1
            semantic_seg[x, y] = margin_left[i]


        return semantic_seg, num_grid

    def pre_eval(self):
        # baseline_dir = self.gt_dir.replace('/gt','/baseline_gt')
        # pred_file_list = os.listdir(baseline_dir)
        gt_file_list = os.listdir(self.gt_dir)
        print("{} files to evaluate".format(len(gt_file_list)))
        count = 0
        result = 0
        for file in tqdm(gt_file_list):
            pred_file = os.path.join(self.pred_dir, file)
            gt_file = os.path.join(self.gt_dir, file)
            if (os.path.isfile(pred_file) == False): 
                continue
            assert os.path.isfile(gt_file), "gt file not exist"
            count += 1

            pred_seg_map, pred_num_grid = self.load_and_transform_txt(pred_file)
            
            compare_seg, num_grid = self.compare_grid(gt_file, pred_seg_map)
            if (num_grid == 0):
                continue
            result += np.sum(compare_seg) / num_grid

            # print("gt_dir: ", gt_dir)
            # print("file: ", file)
            # mask = pred_seg_map != gt_seg_map
            # indices = np.argwhere(mask) + [DETECTION_AREA_EXTENTS[0][0], DETECTION_AREA_EXTENTS[1][0]]
            # print(indices)
            

            # if (pred_num_grid == 0 and gt_num_grid == 0): 
            #     continue
            # result += np.sum(abs(pred_seg_map - gt_seg_map) / (max(pred_num_grid, gt_num_grid)))
        
            

        return result / count

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
        #if isinstance(metric, list):
        #    metric = metric[0]
        if isinstance(metric, str):
            metric = metric.split(',')  # [metric]
        # if isinstance(metric, str):
        #    metric = [metric]
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

    pred_root = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/ray_val/"
    bag_list = sorted(os.listdir(pred_root))
    gt_root = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/ray_gt/"
    class_names = ['background', 'obstacle']
    all_iou = []
    all_presicion = []
    all_recall = []
    all_f1 = []

    for bag in bag_list:
        print("==>evaluating bag:{}".format(bag))
        pred_dir = os.path.join(pred_root, bag, "gt")
        gt_dir = os.path.join(gt_root, bag, "gt")
        custom_eval = Custom_Eval(pred_dir, gt_dir, class_names, )
        pre_eval_results = custom_eval.pre_eval()
        print('margin_right mean: ', pre_eval_results)
        
