# encoding: utf-8
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""

import argparse
import os
import torch
from torch import nn
from torch.autograd import Variable
import time
from torchvision import transforms
from torch.backends import cudnn
from torch.utils.data import DataLoader
import logging

from config import cfg
from dataloader import MyDataset
from modeling import build_model

import json
from tqdm import tqdm
from PIL import Image

root = 'test_list.json'
check = True

def test(cfg):
    # transform
    transform_test_list = [
        transforms.Resize(size=cfg.INPUT.SIZE_TEST, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # prepare dataset
    test_dataset = MyDataset(root=root, transform=transforms.Compose(transform_test_list), type='test')
    test_loader = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=False)
    num_classes = cfg.MODEL.HEADS.NUM_CLASSES

    # prepare model
    def load_network(network):
        save_path = cfg.LOAD_FROM
        checkpoint = torch.load(save_path)
        if 'model' in checkpoint:
            network.load_state_dict(checkpoint['model'])
        else:
            network.load_state_dict(checkpoint)
        return network

    model = build_model(cfg, num_classes)
    model = load_network(model)
    model = model.cuda()

    # for data in tqdm(test_loader):
    for data in test_loader:
        model.train(False)
        inputs, labels = data
        # print(inputs.shape)
        inputs = Variable(inputs.cuda().detach())
        with torch.no_grad():
            out = model(inputs)
            # print(logits)
            score, preds = torch.max(out['pred_class_logits'], 1)
            # print(score, preds)
            if preds.int() == 0:
                cat = "No"
            elif preds.int() == 1:
                cat = "Yes"
            print(preds.cpu().numpy().item(), labels.cpu().numpy().item())

    return


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    test(cfg)


if __name__ == '__main__':
    main()
