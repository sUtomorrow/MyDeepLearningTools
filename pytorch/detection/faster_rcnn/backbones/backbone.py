# -*- coding: utf-8 -*-
# @Time     : 5/25/19 4:09 PM
# @Author   : lty
# @File     : backbone

import torch

class Backbone(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Backbone, self).__init__()

    def forward(self, *input):
        raise NotImplementedError('forward method not implemented')

    @property
    def feature_levels(self):
        """return the downsample times of model's outputs
        """
        return self._feature_levels

    def load_pretrain(self, model_dir):
        """load pretrain model
        """
        raise NotImplementedError('load_pretrain method not implemented.')

    # @staticmethod
    # def image_preprocess_func(backbone_name):
    #     """Takes as input an image and prepares it for being passed through the network.
    #     Having this function in Backbone allows other backbones to define a specific preprocessing step.
    #     """
    #     raise NotImplementedError('image_preprocess method not implemented.')