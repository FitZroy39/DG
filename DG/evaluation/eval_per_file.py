import os
import numpy as np
from metrics import intersect_and_union, pre_eval_to_metrics_per_file, pre_eval_to_metrics
from collections import OrderedDict
from prettytable import PrettyTable
from tqdm import tqdm
import sys
sys.path.append("..") 
from configs import IMG_SIZE, DETECTION_AREA_EXTENTS

class Eval_Per_File:
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
        # is_valid = data[:, 4]
        for i in range(length):
            if is_obstacle[i] == 0:
            # if is_obstacle[i] == 0 or is_valid[i] == 1:
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



    def pre_eval_per_file(self):
        pred_file_list = os.listdir(self.gt_dir)
        print("{} files to evaluate".format(len(pred_file_list)))
        pre_eval_results = []
        file_names = []
        for file in tqdm(pred_file_list):
            pred_file = os.path.join(self.pred_dir, file)
            gt_file = os.path.join(self.gt_dir, file)
            assert os.path.isfile(gt_file), "gt file not exist"

            pred_seg_map = self.load_and_transform_txt(pred_file)
            gt_seg_map = self.load_and_transform_txt(gt_file)

            pre_eval_results.append(intersect_and_union(pred_seg_map, gt_seg_map, self.num_classes,self.ignore_index, self.label_map,
                                    self.reduce_zero_label))
            file_names.append(file)

        return pre_eval_results, file_names

    def evaluate_per_file(self, results, file_names, metric='mIoU', logger=None, **kwargs):
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
            metric = metric.split(',') 

        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        iou_sum = 0
        fscore_sum = 0
        precision_sum = 0
        recall_sum = 0
        for i in range(len(file_names)):
            file_name = file_names[i]
            result = results[i]
            ret_metrics = pre_eval_to_metrics_per_file(result, metric, nan_to_num=0)
            iou = round(ret_metrics['IoU'][1], 3)
            fscore = round(ret_metrics['Fscore'][1], 3)
            precision = round(ret_metrics['Precision'][1], 3)
            recall = round(ret_metrics['Recall'][1], 3)
            iou_sum += iou
            fscore_sum += fscore
            precision_sum += precision
            recall_sum += recall

            print("{}: iou:{}, precision:{}, recall:{}".format(file_name, iou, precision, recall))

        print('-------------------mean iou of all test images-----------------')
        mean_iou = iou_sum / len(file_names)
        mean_fs = fscore_sum / len(file_names)
        mean_ps = precision_sum / len(file_names)
        mean_rc = recall_sum / len(file_names)
        print('mean iou:', mean_iou)
        print('mean fscore:', mean_fs)
        print('mean precision:', mean_ps)
        print('mean recall:', mean_rc)

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

        return

if __name__ == "__main__":
    pred_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/check_new/howo11_2021_05_23_16_29_54_179/gt"
    gt_dir = "/mnt/yrfs/yanrong/pvc-80688cb9-3d14-45f4-9be0-f37238d68d83/personal/hefangzheng/data/grid_benckmark0808_300x300/check/howo11_2021_05_23_16_29_54_179/gt"
    class_names = ['background', 'obstacle']
    eval_per_file = Eval_Per_File(pred_dir, gt_dir, class_names,)
    pre_eval_results, file_names = eval_per_file.pre_eval_per_file()
    eval_results = eval_per_file.evaluate_per_file(pre_eval_results, file_names, metric=['mIoU', 'mDice', 'mFscore'])