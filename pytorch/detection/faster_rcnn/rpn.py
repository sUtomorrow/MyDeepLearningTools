# -*- coding: utf-8 -*-
# @Time     : 9/15/19 9:49 AM
# @Author   : lty
# @File     : rpn

import torch
from .utils.losses import smooth_l1, cross_entropy_loss
from .utils.anchors import get_anchors, anchor_shift


def _bbox_transform(anchors, gt_boxes):
    """
    transform gt boxes to anchor regression targets
    :param anchors : [x1, y1, x2, y2]
    :param gt_boxes: [bx1, by1, bx2, by2]
    ax = (x1 + x2) / 2
    ay = (y1 + y2) / 2
    aw = x2 - x1
    ah = y2 - y1

    bx = (bx1 + bx2) / 2
    by = (by1 + by2) / 2
    bw = bx2 - bx1
    bh = by2 - by1

    tx = (bx - ax) / aw
    ty = (by - ay) / ah
    tw = log(bw / aw)
    th = log(bh / ah)

    return [tx, ty, tw, th]
    """
    ax = (anchors[:, 0] + anchors[:, 2]) / 2
    ay = (anchors[:, 1] + anchors[:, 3]) / 2
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    bx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
    by = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
    bw = gt_boxes[:, 2] - gt_boxes[:, 0]
    bh = gt_boxes[:, 3] - gt_boxes[:, 1]

    tx = (bx - ax) / aw
    ty = (by - ay) / ah
    tw = torch.log(bw / aw)
    th = torch.log(bh / ah)

    return torch.stack([tx, ty, tw, th], axis=-1)

def _bbox_transform_inv(anchors, regressions):
    """
    transform regression back to bbox by anchors
    :param anchors:    [x1, y1, x2, y2]
    :param regressions: [tx, ty, tw, th]
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

    bbox_x = anchor_x + regressions[:, :, 0] * anchor_w
    bbox_y = anchor_y + regressions[:, :, 1] * anchor_h
    bbox_w = anchor_w * torch.exp(regressions[:, :, 2])
    bbox_h = anchor_h * torch.exp(regressions[:, :, 3])

    bbox_x1 = bbox_x - bbox_w / 2
    bbox_y1 = bbox_y - bbox_h / 2
    bbox_x2 = bbox_x1 + bbox_w
    bbox_y2 = bbox_y1 + bbox_h

    return torch.stack([bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=-1)


def overlaps(bboxes1, bboxes2):
    """
    :param bboxes1: [x1, y1, x2, y2] with shape:(M, 4)
    :param bboxes2: [x1, y1, x2, y2] with shape:(N, 4)
    :return: overlaps with shape: (M, N)
    """
    N = bboxes1.size(0)
    M = bboxes2.size(0)

    # reshape for broadcast
    bboxes1 = torch.repeat(torch.unsqueeze(bboxes1, 1), (1, N, 1))
    bboxes2 = torch.repeat(torch.unsqueeze(bboxes2, 0), (M, 1, 1))

    inter_x1 = torch.max(bboxes1[:, :, 0], bboxes2[:, :, 0])
    inter_x2 = torch.min(bboxes1[:, :, 2], bboxes2[:, :, 2])

    inter_y1 = torch.max(bboxes1[:, :, 1], bboxes2[:, :, 1])
    inter_y2 = torch.min(bboxes1[:, :, 3], bboxes2[:, :, 3])

    inter_w = inter_x2 - inter_x1
    inter_w[inter_w < 0] = 0
    inter_h = inter_y2 - inter_y1
    inter_h[inter_h < 0] = 0

    inter_area = inter_h * inter_w

    w1 = bboxes1[:, :, 2] - bboxes1[:, :, 0]
    w1[w1 < 0] = 0
    h1 = bboxes1[:, :, 3] - bboxes1[:, :, 1]
    h1[h1 < 0] = 0

    w2 = bboxes2[:, :, 2] - bboxes2[:, :, 0]
    w2[w2 < 0] = 0
    h2 = bboxes2[:, :, 3] - bboxes2[:, :, 1]
    h2[h2 < 0] = 0

    area1 = w1 * h1
    area2 = w2 * h2

    return inter_area / (area1 + area2 - inter_area)


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
        """
        :param input:
            anchors        : with shape (A, 4), (x1, y1, x2, y2), the anchor for each point
            batch_gt_boxes : with shape (BatchSize, max_target_number, 4), (x1, y1, x2, y2)
            batch_labels   : with shape (BatchSize, max_target_number), contain the label id for boxes
        :return: [(A, 5), (A, 2)] regression target and classification target
        """
        anchors, batch_gt_boxes, batch_labels = input
        #TODO: to compute the target for rpn
        batch_size = batch_gt_boxes.size(0)
        regression_target     = torch.zeros((batch_size, anchors.size(0), 5)) # [tx, ty, tw, th, weight]
        classification_target = torch.zeros((batch_size, anchors.size(0), 2)) # [background, foreground, weight]
        for batch_idx in range(batch_size):
            # compute target for each batch data
            gt_boxes = batch_gt_boxes[batch_idx]
            labels   = batch_labels[batch_idx]

            # filter the padding label and box
            indices  = labels >= 0
            gt_boxes = gt_boxes[indices]
            labels   = labels[indices]

            gt_box_number = labels.size(0)

            if gt_box_number == 0:
                # skip if there is no target
                continue

            ious = overlaps(anchors, gt_boxes)

            max_anchor_ious, anchor_max_iou_gt_box_indices = torch.max(ious, 1)

            positive_anchor_mask    = max_anchor_ious >= self.anchor_positive_threshold
            negative_anchor_mask    = max_anchor_ious <= self.anchor_negative_threshold
            positive_anchor_indices = torch.nonzero(positive_anchor_mask)
            negative_anchor_indices = torch.nonzero(negative_anchor_mask)
            # ignore_anchor_indices   = torch.nonzero(~(positive_anchor_mask | negative_anchor_mask))

            if positive_anchor_indices.size(0) > self.max_positive_anchors:
                # random choice to filter the positive anchor with max positive number
                indices = torch.randperm(positive_anchor_indices.size(0))
                positive_anchor_indices = positive_anchor_indices[indices[:self.max_positive_anchors]]

            if negative_anchor_indices.size(0) > self.max_negative_anchors:
                # random choice to filter the negative anchor with max negative number
                indices = torch.randperm(negative_anchor_indices.size(0))
                negative_anchor_indices = negative_anchor_indices[indices[:self.max_negative_anchors]]

            # compute all anchor regression targets
            anchor_regression_target = _bbox_transform(anchors, gt_boxes[anchor_max_iou_gt_box_indices])

            # assign weight for anchors, default 0 for ignore this anchor
            regression_target[batch_idx, positive_anchor_mask, -1] = 1

            classification_target[batch_idx, positive_anchor_mask, -1] = 1
            classification_target[batch_idx, negative_anchor_indices, -1] = 1

            # assign regression target and classification target
            regression_target[batch_idx, positive_anchor_indices, :-1] = anchor_regression_target[positive_anchor_indices, :]
            classification_target[batch_idx, negative_anchor_indices, 0] = 1  # rpn use label idx 1 as classification target for foreground
        return [regression_target, classification_target]
        

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
        # self.relu = torch.nn.ReLU(inplace=True)
        # background and foreground
        self.classification = torch.nn.Conv2d(filters, anchor_num * 2, kernel_size=(1, 1), padding=(1, 1), stride=(1, 1))
        # self.sigmoid = torch.nn.Sigmoid()
        # 4 values for each anchor
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
        # classification = self.sigmoid(classification)
        regression     = self.regression(l)
        # regression     = self.relu(regression)
        
        bboxes = _bbox_transform_inv(anchors, regression)
        
        if self.training:
            regression_target, classification_target = self.anchor_target_layer(anchors, batch_gt_boxes, batch_labels)
            regression_loss     = smooth_l1(regression, regression_target)
            classification_loss = cross_entropy_loss(classification, classification_target)
            return classification, bboxes, regression_loss, classification_loss
        else:
            return classification, bboxes, 0, 0