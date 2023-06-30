import os

import torch
import pytorch_lightning as pl
from .dataloader import get_transform
from .pc_bev import PC_BEV_Dataset
from .pointpillar import POINTPILLAR_Dataset



class PCBEVDataModule(pl.LightningDataModule):

    def __init__(self, config=None):
        super().__init__()

        self.config = config
        root_path = config['root_path']

        train_dir = os.path.join(root_path, config['train_dir'])
        train_lists = os.listdir(train_dir)
        train_lists = [os.path.join(train_dir,item) for item in train_lists]

        self.dataset = PC_BEV_Dataset(train_lists, 'Train', get_transform(train=True), config)
        print('Train set lenth:', len(self.dataset))

        val_dir = os.path.join(root_path, config['val_dir'])
        val_lists = os.listdir(val_dir)
        val_lists = [os.path.join(val_dir, item) for item in val_lists]

        self.dataset_val = PC_BEV_Dataset(val_lists, 'Val', get_transform(train=False), config)
        print('Val set lenth:', len(self.dataset_val))

    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.config['train_batch_size'], shuffle=False,
            num_workers=self.config['train_num_workers'])
        return data_loader

    def val_dataloader(self):
        dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val, batch_size=self.config['val_batch_size'],
            shuffle=False, num_workers=self.config['val_num_workers'])
        return dataloader_val

import numpy as np
from collections import defaultdict
def merge_second_batch(batch_list, _unused=False):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    #example_merged.pop("num_voxels")
    for key, elems in example_merged.items():
        if key in [
                'voxels', 'num_points'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret

class POINTPILLARDataModule(pl.LightningDataModule):

    def __init__(self, config=None):
        super().__init__()

        self.config = config
        root_path = config['root_path']

        train_dir = os.path.join(root_path, config['train_dir'])
        train_lists = os.listdir(train_dir)
        train_lists = [os.path.join(train_dir,item) for item in train_lists]

        self.dataset = POINTPILLAR_Dataset(train_lists, 'Train', get_transform(train=True), config)
        print('Train set lenth:', len(self.dataset))

        val_dir = os.path.join(root_path, config['val_dir'])
        val_lists = os.listdir(val_dir)
        val_lists = [os.path.join(val_dir, item) for item in val_lists]

        self.dataset_val = POINTPILLAR_Dataset(val_lists, 'Val', get_transform(train=False), config)
        print('Val set lenth:', len(self.dataset_val))

    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.config['train_batch_size'], shuffle=False,
            num_workers=self.config['train_num_workers'],
        collate_fn=merge_second_batch,)
        return data_loader

    def val_dataloader(self):
        dataloader_val = torch.utils.data.DataLoader(
            self.dataset_val, batch_size=self.config['val_batch_size'],
            shuffle=False, num_workers=self.config['val_num_workers'],
        collate_fn=merge_second_batch,)
        return dataloader_val