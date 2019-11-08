# -*- coding: utf-8 -*-
# @Time     : 9/16/19 9:36 PM
# @Author   : lty
# @File     : config

valid_backbone = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

class Config(object):

    num_classes             = 1 + 1             # 0 is background
    fpn                     = True
    input_image_size        = (512, 512)        # the input size
    backbone_name           = 'resnet101'
    backbone_feature_channel=  1024 # 2048
    backbone_pretrain       = True              # load the backbone model weights pre-train on ImageNet
    anchor_sizes            = [[64, 128, 256], ] # the base anchor sizes
    anchor_ratios           = [0.5, 1., 2.]    # the ratios of w:h for each anchor size
    roi_size                = (14, 14)           # output size of roi pooling or roi align
    roi_align               = True             # if use roi align
    roi_channel             = 2048
    use_feature_levels      = [4,] # 5             # features levels to use, only support 5
    rpn_filters             = 256              # the output channel of conv1 in rpn
    rpn_score_threshold     = 0.5              # to filter rpn bboxes when doing roi proposal

    positive_roi_bbox_threshold = 0.5
    negative_roi_bbox_threshold = 0.3
    max_positive_roi_bbox       = 100
    max_negative_roi_bbox_ratio = 0.5


    positive_anchor_threshold = 0.5         # iou threshold to decide an anchor is positive(greater or equal)
    negative_anchor_threshold = 0.3         # iou threshold to decide an anchor is negative(less or equal)
    max_positive_anchor_num   = 100         # max number of positive anchor in one image, if None: no limit
    max_negative_anchor_ratio = 0.5         # max ratio, negative anchor number : positive anchor number, if None: no limit
    max_outputs_per_image     = 100         # max output bboxes

    def __init__(self, **kwargs):
        print(kwargs)
        for key in kwargs:
            self.__setattr__(key, kwargs[key])

        self.anchor_num = len(self.anchor_ratios) * len(self.anchor_sizes[0])

        self.check_params()

        if 'resnet18' in self.backbone_name:
            self.backbone_feature_channel = 256
            self.roi_channel              = 512
        elif 'resnet' in self.backbone_name:
            self.backbone_feature_channel = 1024
            self.roi_channel              = 2048


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
        assert (self.max_outputs_per_image is None or self.max_outputs_per_image > 0,
                'the max_outputs_per_image should be positive number, but get max_target_num_per_image={}'.format(
                    self.max_outputs_per_image))

        if not self.fpn:
            pass