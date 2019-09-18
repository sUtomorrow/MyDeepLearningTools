# -*- coding: utf-8 -*-
# @Time     : 9/15/19 9:49 AM
# @Author   : lty
# @File     : rpn

import torch
from .utils.anchors import get_anchors, anchor_shift

def _bbox_transform_inv(anchors, regression):
    """
    transform regression back to bbox by anchors
    :param anchors:    [x1, y1, x2, y2]
    :param regression: [tx, ty, tw, th]
    ax = (x1 + x2) / 2
    ay = (y1 + y2) / 2
    aw = x2 - x1
    ah = y2 - y1

    bx = ax + tx * aw
    by = ay + ty * ah
    bw = aw * exp(tw)
    bh = ah * exp(th)

    bx1 = bx - bw / 2
    by1 = by - bh / 2
    bx2 = bx1 + bw
    by2 = by1 + bh
    return [bx1, by1, bx2, by2]
    """

    anchor_x = (anchors[:, :, 2] + anchors[:, :, 0]) / 2
    anchor_y = (anchors[:, :, 3] + anchors[:, :, 1]) / 2

    anchor_w = anchors[:, :, 2] - anchors[:, :, 0]
    anchor_h = anchors[:, :, 3] - anchors[:, :, 1]

    bbox_x = anchor_x + regression[:, :, 0] * anchor_w
    bbox_y = anchor_y + regression[:, :, 1] * anchor_h
    bbox_w = anchor_w * torch.exp(regression[:, :, 2])
    bbox_h = anchor_h * torch.exp(regression[:, :, 3])

    bbox_x1 = bbox_x - bbox_w / 2
    bbox_y1 = bbox_y - bbox_h / 2
    bbox_x2 = bbox_x1 + bbox_w
    bbox_y2 = bbox_y1 + bbox_h

    return torch.stack([bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=-1)


class AnchorTargetLayer(torch.nn.Module):
    def __init__(self, anchor_positive_threshold, anchor_negative_threshold, max_positive_anchors, max_negative_anchors):
        """
        compute the rpn target by anchor and feature map

        :param anchor_positive_threshold: iou threshold to decide an anchor is positive(greater or equal)
        :param anchor_negative_threshold: iou threshold to decide an anchor is negative(less or equal)
        :param max_positive_anchors:      max number of positive anchor in one image, if None: no limit
        :param max_negative_anchors:      max number of negative anchor in one image, if None: no limit
        """
        super(AnchorTargetLayer, self).__init__()

        self.anchor_positive_threshold = anchor_positive_threshold
        self.anchor_nagetive_threshold = anchor_negative_threshold

        self.max_positive_anchors = max_positive_anchors
        self.max_negative_anchors = max_negative_anchors

    def forward(self, *input):
        anchors, batch_gt_boxes, batch_labels = input
        #TODO: to compute the target for rpn



class RPN(torch.nn.Module):
    def __init__(self, in_channel, filters, anchor_num,
                 anchor_positive_threshold, anchor_negative_threshold, max_positive_anchors, max_negative_anchors):
        """
        the rpn model for faster rcnn
        :param in_channel:                the channel of input feature map
        :param filters:                   the channel of convolution layer in rpn
        :param anchor_num:                amount of anchors for each point
        :param anchor_positive_threshold: iou threshold to decide an anchor is positive(greater or equal)
        :param anchor_negative_threshold: iou threshold to decide an anchor is negative(less or equal)
        :param max_positive_anchors:      max number of positive anchor in one image, if None: no limit
        :param max_negative_anchors:      max number of negative anchor in one image, if None: no limit
        """
        super(RPN, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channel, filters, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
        self.classification = torch.nn.Conv2d(filters, anchor_num, kernel_size=(1, 1), padding=(1, 1), stride=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()
        self.regression = torch.nn.Conv2d(filters, anchor_num * 4, kernel_size=(1, 1), padding=(1, 1), stride=(1, 1))
        self.anchor_target_layer = AnchorTargetLayer(anchor_positive_threshold,
                                                     anchor_negative_threshold,
                                                     max_positive_anchors,
                                                     max_negative_anchors)

    def forward(self, *input):
        if self.training:
            feature, anchors, batch_gt_boxes, batch_labels = input
        else:
            batch_gt_boxes = batch_labels = None
            feature, anchors = input[:2]
        l = self.conv1(feature)
        l = self.relu(l)
        classification = self.classification(l)
        classification = self.sigmoid(classification)
        regression     = self.regression(l)
        regression     = self.relu(regression)
        
        bboxes = _bbox_transform_inv(anchors, regression)
        
        if self.training:
            rpn_target = self.anchor_target_layer(anchors, batch_gt_boxes, batch_labels)
            #TODO: to get the loss of classification and regression
        else:
            return classification, bboxes, 0, 0