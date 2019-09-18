# -*- coding: utf-8 -*-
# @Time     : 9/16/19 9:36 PM
# @Author   : lty
# @File     : config

class Config(object):
    anchor_sizes         = [64, 128, 256]   # the base anchor sizes
    anchor_ratios        = [0.5, 1., 2.]    # the ratios of w:h for each anchor size
    roi_pooling_size     = (6, 6)           # output size of roi pooling or roi align
    roi_align            = True             # if use roi align
    multi_feature_levels = False            # the fpn is not supported now, should be False
    feature_levels       = [5]

    anchor_positive_threshold = 0.5         # iou threshold to decide an anchor is positive(greater or equal)
    anchor_negative_threshold = 0.2         # iou threshold to decide an anchor is negative(less or equal)
    anchor_max_positive_num   = 100         # max number of positive anchor in one image, if None: no limit
    anchor_max_nagetive_num   = 300         # max number of negative anchor in one image, if None: no limit


    def __init__(self, **kwargs):
        for key, value in kwargs:
            self.__setattr__(key, value)

        self.anchor_num = len(self.anchor_ratios) * len(self.anchor_sizes)
        self.check_params()

    def check_params(self):
        assert (self.anchor_max_positive_num is None or self.anchor_max_positive_num > 0,
               'the anchor_max_positive_num should be positive number or None, but get anchor_max_positive_num={}'.format(
                   self.anchor_max_positive_num))
        assert (self.anchor_negative_threshold is None or self.anchor_negative_threshold > 0,
                'the anchor_nagetive_threshold should be positive number or None, but get anchor_nagetive_threshold={}'.format(
                    self.anchor_negative_threshold))
