import os
import numpy as np
from .transform import Compose, ToTensor
from imageio import imread
from glob import glob

import torch
from torch import nn
import torch.utils.data
import random

from random import shuffle

def get_img_list(file_path):
    "Read lines from a txt file"
    paths = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip()
            paths.append(line)
    return paths

def remove_from_list(img_paths, exclude_paths):
    img_paths = set(img_paths)
    img_paths.difference_update(set(exclude_paths))
    return list(img_paths)


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
    return Compose(transforms)




