# -*- coding: utf-8 -*-
# @Time     : 9/16/19 9:35 PM
# @Author   : lty
# @File     : frcnn

import os
import sys

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import faster_rcnn  # noqa: F401
    __package__ = "faster_rcnn"

import torch
from .backbones import *
from .rpn import RPN
from .utils.anchors import get_anchors, anchor_shift
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

        if config.backbone_pretrain:
            self.backbone.load_pretrain()

        self.backbone_feature_levels = self.backbone.feature_levels

        self.use_feature_levels = config.use_feature_levels

        self.use_feature_idx_list = []

        # find the feature index in backbone model's output
        for feature_level in self.use_feature_levels:
            if feature_level in self.backbone_feature_levels:
                self.use_feature_idx_list.append(self.backbone_feature_levels.index(feature_level))
            else:
                raise ValueError('backbone:{} do not support feature level:{}'.format(config.backbone_name, feature_level))

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
            # input should contain gt boxes and labels in training mode
            batch_image, batch_gt_boxes, batch_labels = input
        else:
            batch_image = input[0]
            batch_gt_boxes = batch_labels = None

        # fit image to backbone model and get the features as output
        backbone_features = self.backbone(batch_image)

        feature_list = []
        anchors_list = []

        # get backbone features for each feature level and anchors for each feature
        for feature_level, feature_idx in zip(self.use_feature_levels, self.use_feature_idx_list):
            use_feature = backbone_features[feature_idx]
            anchors = get_anchors(anchor_sizes=self.anchor_sizes, anchor_ratios=self.anchor_ratios)
            anchors = anchor_shift(anchors, use_feature.size()[2:4], stride=2 ** feature_level)
            anchors = torch.from_numpy(anchors).float()
            if torch.cuda.is_available():
                anchors = anchors.cuda()
            feature_list.append(use_feature)
            anchors_list.append(anchors)

        if self.training:
            rpn_bboxes_list              = []
            rpn_scores_list              = []
            rpn_regression_loss_list     = []
            rpn_classification_loss_list = []
            for feature, anchors in zip(feature_list, anchors_list):
                # use rpn to get proposal bbox for each feature level
                rpn_bboxes, rpn_scores, rpn_regression_loss, rpn_classification_loss = self.rpn(
                    feature, anchors, batch_gt_boxes, batch_labels)
                rpn_bboxes_list.append(rpn_bboxes)
                rpn_scores_list.append(rpn_scores)
                rpn_regression_loss_list.append(rpn_regression_loss)
                rpn_classification_loss_list.append(rpn_classification_loss)

            # total loss is the average of losses in each feature level
            rpn_regression_loss     = torch.stack(rpn_regression_loss_list).mean()
            rpn_classification_loss = torch.stack(rpn_classification_loss_list).mean()
            return rpn_bboxes_list, rpn_scores_list, rpn_regression_loss, rpn_classification_loss
        else:
            rpn_bboxes_list = []
            rpn_scores_list = []
            for feature, anchors in zip(feature_list, anchors_list):
                # use rpn to get proposal bbox for each feature level
                # there is no rpn loss for eval mode
                rpn_bboxes, rpn_classification = self.rpn(feature, anchors, batch_gt_boxes, batch_labels)[:2]
                rpn_bboxes_list.append(rpn_bboxes)
                rpn_scores_list.append(rpn_classification)
            return rpn_bboxes_list, rpn_scores_list


if __name__ == '__main__':
    """test Faster Rcnn with only rpn"""

    # print(__package__)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    device = torch.device('cuda:0')

    test_config = Config()
    frcnn = FasterRcnn(test_config)

    # frcnn = torch.nn.DataParallel(frcnn)

    frcnn.to(device)

    frcnn.eval()

    # print(frcnn.rpn.regression.weight)

    print('start run')

    for i in range(3):
        batch_image = torch.rand((2, 3, 64, 64))
        # print(batch_image.mean())
        batch_image = batch_image.to(device)
        print('iter:', i, frcnn(batch_image))

    # print(frcnn)




