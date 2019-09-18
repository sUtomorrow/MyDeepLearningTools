# -*- coding: utf-8 -*-
# @Time     : 9/16/19 9:35 PM
# @Author   : lty
# @File     : frcnn

import torch
from .backbones import *
from .rpn import RPN
from .config import Config

class FasterRcnn(torch.nn.Module):
    def __init__(self, config):
        """
        :param config:
            feature_levels: list, the feature levels to proposal bbox
        """
        super(FasterRcnn, self).__init__()
