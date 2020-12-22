#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import logging
from PIL import Image
import numpy as np
import glob
import os
from tabulate import tabulate
import json

class MyDataset(Dataset):
    def __init__(self, root=None, transform=None, type='train'):
        self.type = type
        with open(root, 'r') as f:
            datalist = json.load(f)

        if self.type == 'train':
            self.datapath = datalist[:int(0.9*len(datalist))]
        elif self.type == 'val':
            self.datapath = datalist[int(0.9 * len(datalist)):]
        elif self.type == 'test':
            self.datapath = datalist
        else:
            RuntimeError

        self.len = len(self.datapath)

        self.transform = transform

        table = [["{}".format(self.type), self.len]]
        headers = ['stage', 'len']
        datainfo = tabulate(table, headers, tablefmt="grid")
        logger = logging.getLogger('merlin.baseline.dataset')
        logger.info('\n' + datainfo)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.datapath[index]['img']
        label = self.datapath[index]['label']

        try:
            img = Image.open(img_path)
        except:
            print(img_path)
            print('Corrupted image for %d' % index)
            return self[index + 1]

        if self.transform is not None:
            img = self.transform(img)

        return (img, int(label))

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        # img.show()
        # resize
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img