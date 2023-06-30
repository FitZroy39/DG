from __future__ import print_function

from collections import defaultdict, deque
import datetime
import pickle
import time
import json

import torch
import torch.distributed as dist
import numpy as np

import errno
import os

import yaml
     

def get_img_list(file_path):
    "Read lines from a txt file"
    paths = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            paths.append(line)
    return paths


def write_to_file(string, file_path, create_file=False, override=False):
    """
        Writing sring or a list of string to a file, 
        file may exist or not,
        can override or append
    """
    assert not ((not os.path.exists(file_path)) and (not create_file)), "File does not exist！"

    token = 'w+' if override else 'a+'
    with open(file_path, token) as f:
        if type(string) == str:
            f.writelines(string+'\n')
        else:
            for line in string:
                f.writelines(line+'\n')
    return None


def load_from_json(json_name):
    with open(json_name, 'r') as fp:
        info = json.load(fp)
    return info

def remove_key(checkpoint, exclude_key_words):
    new_dict = {}
    for key in checkpoint:
        flag = False
        for exclude_key_word in exclude_key_words:
            if exclude_key_word in key:
                print('Removed key:', key)
                flag = True
        if not flag:
            new_dict[key] = checkpoint[key]
    return new_dict




def model_speed_test(model, repeat = 100, observability=False):
    import time
    if observability:
        from eval_bev_with_observability import read_pcd, AREA_EXTENTS, SELF_CAR_EXTENTS, num_slice, voxel_size, \
            log_norm, F, ExtractBevFeature, cal_observabilty, cal_grid_pcd
    else:
        from eval_bev import read_pcd, AREA_EXTENTS, SELF_CAR_EXTENTS, num_slice, voxel_size, log_norm, F, \
            ExtractBevFeature
    load_times = []
    transform_times = []
    for i in range(repeat):
        #pcd_path = "/private/personal/linyuqi/grid_benchmark0503/howo10_2021_05_01_12_52_39_5/pcd/1619844760.919672.pcd"
        pcd_path = '/root/grid_benchmark0503_300x300/grid_benchmark0503_300x300_val/howo20_2021_11_20_10_52_30_124/pcd/1637376793.760042.pcd'
        start1 = time.time()
        if pcd_path.endswith('.txt'):
            points = np.loadtxt(pcd_path)
        else:
            points = read_pcd(pcd_path)
        load_times.append((time.time()-start1)*1000)
        start2 = time.time()
        extract_bev = ExtractBevFeature(area_extents=AREA_EXTENTS,
                                        self_car_extents=SELF_CAR_EXTENTS,
                                        num_slices=num_slice,
                                        voxel_size=voxel_size,
                                        log_norm=log_norm)
        bev_img = extract_bev(points)
        if observability:
            grid_pcd_cnt = cal_grid_pcd(points)
            obs = cal_observabilty(points, grid_pcd_cnt)
            max_num = np.max(obs)
            # obs[obs > max_num] = max_num
            obs = obs / max_num
            bev_img = np.concatenate((bev_img, obs), axis=-1)
        img = F.to_tensor(bev_img).cuda().unsqueeze(0)
        transform_times.append((time.time() - start2) * 1000)
    
    model.eval()
    times = []
    with torch.no_grad():
        for i in range(repeat):
            start = time.time()
            out = model(img)
            torch.cuda.synchronize()
            times.append((time.time()-start)*1000)
    print('load Speed: {} ms.'.format(np.mean(load_times)))
    print('transform Speed: {} ms.'.format(np.mean(transform_times)))
    print('inference Speed: {} ms.'.format(np.mean(times)))





def model_speed_test_pillar(model, repeat=100):
    import time
    from eval_pillar import read_pcd, AREA_EXTENTS, SELF_CAR_EXTENTS, voxel_size, F, VoxelGenerator, point_cloud_range
    load_times = []
    transform_times = []
    for i in range(repeat):
        pcd_path = "/private/personal/linyuqi/gpu12/mmsegmentation/data/pc_bev_rain/images/validation/1619943491.090375.pcd"
        start1 = time.time()
        if pcd_path.endswith('.txt'):
            points = np.loadtxt(pcd_path)
        else:
            points = read_pcd(pcd_path)
        load_times.append((time.time() - start1) * 1000)
        start2 = time.time()
        voxel_gen = VoxelGenerator(voxel_size,
                                   point_cloud_range,
                                   max_num_points=100,
                                   max_voxels=40000,  # 12000,
                                   self_car_extend=SELF_CAR_EXTENTS)
        voxels, coors, num_points_per_voxel = voxel_gen.generate(points, max_voxels=40000)  # 12000)
        coors = np.pad(
            coors, ((0, 0), (1, 0)),
            mode='constant',
            constant_values=0)
        #print(voxels.shape, coors.shape, num_points_per_voxel.shape)
        #voxels = F.to_tensor(voxels).cuda()#.unsqueeze(0)
        #coors = F.to_tensor(coors).cuda()#.unsqueeze(0)
        #num_points_per_voxel = F.to_tensor(num_points_per_voxel).cuda()#.unsqueeze(0)
        data = {}
        data['voxels'] = voxels
        data['coordinates'] = coors
        data['num_points'] = num_points_per_voxel
        #data['targets'] = None

        transform_times.append((time.time() - start2) * 1000)

    model.eval()
    times = []
    with torch.no_grad():
        for i in range(repeat):
            start = time.time()
            out = model(data)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
    print('load Speed: {} ms.'.format(np.mean(load_times)))
    print('transform Speed: {} ms.'.format(np.mean(transform_times)))
    print('inference Speed: {} ms.'.format(np.mean(times)))

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

    
class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {}'.format(header, total_time_str))

