# encoding: utf-8
"""
@author:  merlin
@contact: merlinarer@gmail.com
"""
import torch

from .baseline import Baseline

def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    model = Baseline(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
