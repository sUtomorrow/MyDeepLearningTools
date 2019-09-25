# -*- coding: utf-8 -*-
# @Time     : 9/16/19 9:36 PM
# @Author   : lty
# @File     : config

valid_backbone = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

class Config(object):
    backbone_name           = 'resnet50'
    backbone_output_channel = 2048
    backbone_pretrain       = True               # load the backbone model weights pre-train on ImageNet
    anchor_sizes            = [64, 128, 256]   # the base anchor sizes
    anchor_ratios           = [0.5, 1., 2.]    # the ratios of w:h for each anchor size
    roi_pooling_size        = (6, 6)           # output size of roi pooling or roi align
    roi_align               = True             # if use roi align
    use_feature_levels      = [5,]             # features levels to use, only support 5
    rpn_filters             = 256              # the output channel of conv1 in rpn

    positive_anchor_threshold = 0.3         # iou threshold to decide an anchor is positive(greater or equal)
    negative_anchor_threshold = 0.1         # iou threshold to decide an anchor is negative(less or equal)
    max_positive_anchor_num   = 100         # max number of positive anchor in one image, if None: no limit
    max_negative_anchor_ratio = 3.0         # max ratio, negative anchor number : positive anchor number, if None: no limit

    max_target_num_per_image  = 100         # to pack the label and box to batch box and batch label, should padding

    def __init__(self, **kwargs):
        for key, value in kwargs:
            self.__setattr__(key, value)

        self.anchor_num = len(self.anchor_ratios) * len(self.anchor_sizes)
        self.check_params()

        if 'resnet18' in self.backbone_name:
            self.backbone_output_channel = 512
        elif 'resnet' in self.backbone_name:
            self.backbone_output_channel = 2048

    def check_params(self):
        assert (self.backbone_name in valid_backbone, 'backbone {} not supported, only support backbone in {}'.format(self.backbone_name, valid_backbone))

        assert (self.max_positive_anchor_num is None or self.max_positive_anchor_num > 0,
               'the anchor_max_positive_num should be positive number or None, but get anchor_max_positive_num={}'.format(
                   self.max_positive_anchor_num))
        assert (self.max_negative_anchor_ratio is None or self.max_negative_anchor_ratio > 0,
                'the anchor_max_negative_ratio should be positive float or None, but get anchor_max_negative_ratio={}'.format(
                    self.max_negative_anchor_ratio))

        assert (self.negative_anchor_threshold is None or self.negative_anchor_threshold > 0,
                'the anchor_nagetive_threshold should be positive number or None, but get anchor_nagetive_threshold={}'.format(
                    self.negative_anchor_threshold))
        assert (self.positive_anchor_threshold is None or self.positive_anchor_threshold > 0,
                'the anchor_positive_threshold should be positive number or None, but get anchor_positive_threshold={}'.format(
                    self.positive_anchor_threshold))
        assert (self.max_target_num_per_image is not None and self.max_target_num_per_image > 0,
                'the max_target_num_per_image should be positive number, but get max_target_num_per_image={}'.format(
                    self.max_target_num_per_image))