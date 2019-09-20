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

        self.anchor_sizes  = config.anchor_sizes
        self.anchor_ratios = config.anchor_ratios

        if 'resnet' in config.backbone_name:
            self.backbone = ResNetBackbone(config.backbone_name)
        else:
            raise NotImplementedError('backbone {} not implemented'.format(config.backbone_name))

        self.rpn = RPN(
            config.backbone_output_channel,
            config.rpn_filters,
            config.anchor_num,
            config.anchor_positive_threshold,
            config.anchor_negative_threshold,
            config.anchor_max_positive_num,
            config.anchor_max_nagetive_num
        )

    def forward(self, *input):
        if self.training:
            self.rpn.train()
            #TODO: implement faster rcnn with only rpn train
        else:
            # TODO: implement faster rcnn with only rpn eval
            self.rpn.eval()