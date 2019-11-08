# -*- coding: utf-8 -*-
# @Time     : 9/15/19 9:49 AM
# @Author   : lty
# @File     : rpn

import torch
import torch.nn.functional as F
from .utils.losses import smooth_l1, cross_entropy_loss, focal_loss
from .utils.extensions import _C
from .utils.anchors import _bbox_transform_inv, AnchorTargetLayer

def get_nms_bboxes(bboxes, classifications, bbox_tags, max_outputs, nms_threshold=0.5):
    with torch.no_grad():

        # print('bboxes.size()', bboxes.size())
        # print('scores.size()', scores.size())
        # print('labels.size()', labels.size())

        batch_size      = bboxes.size(0)
        proposal_bboxes = torch.zeros(batch_size, max_outputs, 4).type_as(bboxes)
        proposal_scores = torch.ones(batch_size, max_outputs).type_as(classifications) * -1
        proposal_labels = torch.ones(batch_size, max_outputs).type_as(bboxes).long() * -1
        for batch_idx in range(batch_size):
            b = bboxes[batch_idx, :, :]
            c = classifications[batch_idx, :, :]
            if bbox_tags is not None:
                b = b[bbox_tags[batch_idx]]
                c = c[bbox_tags[batch_idx]]

            if c.size(0) == 0:
                continue

            classifications_softmax = F.softmax(c, dim=1)
            # print('classifications_softmax.size()', classifications_softmax.size())
            s, l = torch.max(classifications_softmax, dim=1)

            mask = l > 0
            b = b[mask, :]
            s = s[mask]
            l = l[mask]

            if nms_threshold is None:
                sorted_indices = torch.argsort(s, dim=0, descending=True)
                # sorted_indices = sorted_indices[:2 * max_outputs]
                b = b[sorted_indices, :]
                s = s[sorted_indices]
                l = l[sorted_indices]
            else:
                indices = _C.nms(b, s, nms_threshold)

                b = b[indices, :]
                s = s[indices]
                l = l[indices]

            proposal_num = l.size(0)
            if proposal_num < max_outputs:
                proposal_bboxes[batch_idx, :proposal_num, :] = b
                proposal_scores[batch_idx, :proposal_num]    = s
                proposal_labels[batch_idx, :proposal_num]    = l
            else:
                proposal_bboxes[batch_idx, :, :] = b[:max_outputs, :]
                proposal_scores[batch_idx, :]    = s[:max_outputs]
                proposal_labels[batch_idx, :]    = l[:max_outputs]
        return proposal_bboxes, proposal_scores, proposal_labels

class RPN(torch.nn.Module):
    def __init__(self, in_channel, filters, anchor_num,
                 positive_anchor_threshold, negative_anchor_threshold, max_positive_anchor, max_negative_anchor_ratio):
        """
        the rpn model for faster rcnn
        :param in_channel:                the channel of input feature map
        :param filters:                   the channel of convolution layer in rpn
        :param anchor_num:                amount of anchors for each point
        :param positive_anchor_threshold: iou threshold to decide an anchor is positive(greater or equal)
        :param negative_anchor_threshold: iou threshold to decide an anchor is negative(less or equal)
        :param max_positive_anchor      : max number of positive anchor in one image, if None: no limit
        :param max_negative_anchor_ratio: max ratio, negative anchor number : positive anchor number, if None: no limit
        """
        super(RPN, self).__init__()
        self.anchor_num = anchor_num
        self.conv1 = torch.nn.Conv2d(in_channel, filters, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
        # background and foreground
        self.classification = torch.nn.Conv2d(filters, self.anchor_num * 2, kernel_size=(1, 1), padding=0, stride=(1, 1))
        # 4 values for each anchor
        self.regression = torch.nn.Conv2d(filters, self.anchor_num * 4, kernel_size=(1, 1), padding=0, stride=(1, 1))
        self.anchor_target_layer = AnchorTargetLayer(positive_anchor_threshold,
                                                     negative_anchor_threshold,
                                                     max_positive_anchor,
                                                     max_negative_anchor_ratio)
        self.max_proposal = max_positive_anchor

    def forward(self, *input):
        """

        :param input:
        :return:
        """
        if self.training:
            feature, anchors, batch_gt_boxes, batch_labels = input
        else:
            batch_gt_boxes   = batch_labels = None
            feature, anchors = input[:2]
        l = self.conv1(feature)
        l = self.relu(l)
        classifications = self.classification(l)
        regressions     = self.regression(l)
        
        # regression     = self.relu(regression)
        regressions     = regressions.permute(0, 2, 3, 1).contiguous().view(regressions.size(0), -1, 4)
        classifications = classifications.permute(0, 2, 3, 1).contiguous().view(classifications.size(0), -1, 2)
        bboxes          = _bbox_transform_inv(anchors.unsqueeze(0), regressions)

        proposal_bboxes, proposal_scores, proposal_labels = get_nms_bboxes(bboxes, classifications, None, self.max_proposal, 0.7)

        if self.training:
            regression_target, classification_target = self.anchor_target_layer(anchors, batch_gt_boxes, batch_labels)
            regression_loss     = smooth_l1(regressions, regression_target, 1)
            classification_loss = focal_loss(classifications, classification_target, 2)
            return proposal_bboxes, proposal_scores, regression_loss, classification_loss
        else:
            return proposal_bboxes, proposal_scores