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
import torch.nn.functional as F
from .backbones import *
from .rpn import RPN, get_nms_bboxes
from .utils.anchors import get_anchors, anchor_shift, _bbox_transform_inv, overlaps, _bbox_transform
from .utils.extensions.roi_align import ROIAlign
from .utils.extensions.roi_pool import ROIPool
from .config import Config
from .utils import losses


class TargetLayer(torch.nn.Module):
    def __init__(self, positive_threshold, negative_threshold, max_positive, max_negative_ratio):
        """
        compute the rpn target by anchor and feature map

        :param positive_threshold: iou threshold to decide an anchor is positive(greater or equal)
        :param negative_threshold: iou threshold to decide an anchor is negative(less or equal)
        :param max_positive     : max number of positive anchor in one image, if None: no limit
        :param max_negative_ratio: max ratio, negative anchor number : positive anchor number, if None: no limit
        """
        super(TargetLayer, self).__init__()

        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

        self.max_positive       = max_positive
        self.max_negative_ratio = max_negative_ratio

    def forward(self, *input):
        """
        :param input:
            batch_roi_bboxes : with shape (BatchSize, N, 4), (x1, y1, x2, y2), the bboxes for each roi
            batch_roi_scores : with shape (BatchSize, N), the scores for each roi, <= 0 for ignore rois
            batch_gt_boxes   : with shape (BatchSize, max_target_number, 4), (x1, y1, x2, y2)
            batch_labels     : with shape (BatchSize, max_target_number), contain the label id for boxes
        :return: [(B, N, 5), (B, N, 2)] regression target and classification target
        """
        with torch.no_grad():
            batch_roi_bboxes, batch_roi_tags, batch_gt_boxes, batch_labels = input
            # print('batch_roi_bboxes.size()', batch_roi_bboxes.size())
            # print('batch_labels.type()', batch_labels.type())
            batch_size = batch_gt_boxes.size(0)
            batch_regression_target     = torch.zeros((batch_size, batch_roi_bboxes.size(1), 5)).type_as(batch_gt_boxes)  # [tx, ty, tw, th, weight]
            batch_classification_target = torch.zeros((batch_size, batch_roi_tags.size(1), 2)).type_as(batch_labels)      # [class_idx, weight]
            for batch_idx in range(batch_size):
                # compute target for each batch data
                gt_boxes   = batch_gt_boxes[batch_idx]
                labels     = batch_labels[batch_idx]
                roi_bboxes = batch_roi_bboxes[batch_idx]
                roi_tags   = batch_roi_tags[batch_idx]
                # filter the padding label and box
                mask     = labels >= 0
                gt_boxes = gt_boxes[mask]
                labels   = labels[mask]
                # print('gt_boxes', gt_boxes)
                # print(gt_boxes[gt_boxes[:, 0] >= gt_boxes[:, 2]])
                # print(gt_boxes[gt_boxes[:, 1] >= gt_boxes[:, 3]])

                # filter padding rois
                roi_mask    = roi_tags
                # roi_bboxes  = roi_bboxes[roi_mask]
                # roi_scores  = roi_scores[roi_mask]


                gt_box_number = labels.size(0)
                roi_number    = roi_tags.size(0)

                if gt_box_number == 0 and roi_number > 0:
                    # random select negative roi if there is no target
                    random_select_negative_mask = (torch.rand(roi_mask.size()) > 0.5) & roi_mask
                    batch_classification_target[batch_idx, random_select_negative_mask, -1] = 1
                    batch_classification_target[batch_idx, random_select_negative_mask, 0]  = 0
                    continue
                elif roi_number == 0:
                    continue
                # print('roi_bboxes.size', roi_bboxes.size())
                # print('gt_boxes.size', gt_boxes.size())
                ious = overlaps(roi_bboxes, gt_boxes)

                # print('ious.size()', ious.size())
                max_roi_ious, roi_max_iou_gt_box_indices = torch.max(ious, 1)
                # print('max_roi_ious.size()', max_roi_ious.size())
                positive_roi_mask = (max_roi_ious >= self.positive_threshold) & roi_mask
                negative_roi_mask = (max_roi_ious <= self.negative_threshold) & roi_mask
                positive_roi_indices = torch.nonzero(positive_roi_mask).view(-1)
                negative_roi_indices = torch.nonzero(negative_roi_mask).view(-1)
                if positive_roi_indices.size(0) == 0 and negative_roi_indices.size(0) == 0:
                    continue
                # ignore_anchor_indices   = torch.nonzero(~(positive_anchor_mask | negative_anchor_mask))

                if self.max_positive:
                    if positive_roi_indices.size(0) > self.max_positive:
                        # random choice to filter the positive anchor with max positive number
                        indices = torch.randperm(positive_roi_indices.size(0))
                        positive_roi_indices = positive_roi_indices[indices[:self.max_positive]]
                    print('max_positive', self.max_positive)

                if self.max_negative_ratio:
                    max_negative = int(self.max_negative_ratio * len(positive_roi_indices))
                    # print('max_negative', max_negative)

                    if negative_roi_indices.size(0) > max_negative:
                        # random choice to filter the negative anchor with max negative number
                        indices = torch.randperm(negative_roi_indices.size(0))
                        negative_roi_indices = negative_roi_indices[indices[:max_negative]]
                    print('max_negative', max_negative)

                # compute all anchor regression targets
                regression_target = _bbox_transform(roi_bboxes, gt_boxes[roi_max_iou_gt_box_indices])

                label_target      = labels[roi_max_iou_gt_box_indices]

                # print('label_target.type()', label_target.type())
                # assign weight for anchors, default 0 for ignore this anchor
                batch_regression_target[batch_idx, positive_roi_indices, -1] = 1

                batch_classification_target[batch_idx, positive_roi_indices, -1] = 1
                batch_classification_target[batch_idx, negative_roi_indices, -1] = 1

                # print('regression_target[positive_roi_indices, :][:2]', regression_target[positive_roi_indices, :][:2])

                # assign regression target and classification target
                batch_regression_target[batch_idx, positive_roi_indices, :-1] = regression_target[positive_roi_indices, :]

                    # exit()

                batch_classification_target[batch_idx, positive_roi_indices, 0] = label_target[positive_roi_indices]

                # if positive_roi_indices.size(0) >= 1:
                # print('roi positive num:', positive_roi_indices.size(0))
                # print('roi negative num:', negative_roi_indices.size(0))
                # print('positive_roi_indices:', positive_roi_indices)
                # print('negative_roi_indices:', negative_roi_indices)
                # print('roi gt    ', gt_boxes[roi_max_iou_gt_box_indices][positive_roi_indices, :][:2])
                # print('roi bboxes', roi_bboxes[positive_roi_indices, :][:2])
                # print('roi target', batch_regression_target[batch_idx, positive_roi_indices, :][:2])
                # print('roi label ', batch_classification_target[batch_idx, positive_roi_indices, :][:2])

            return batch_regression_target, batch_classification_target


class FasterRcnn(torch.nn.Module):
    def __init__(self, config):
        """
        :param config:
            feature_levels: list, the feature levels to proposal bbox
        """
        super(FasterRcnn, self).__init__()

        self.anchor_sizes  = config.anchor_sizes
        self.anchor_ratios = config.anchor_ratios
        self.num_classes   = config.num_classes

        self.max_outputs_per_image = config.max_outputs_per_image

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
            config.backbone_feature_channel,
            config.rpn_filters,
            config.anchor_num,
            config.positive_anchor_threshold,
            config.negative_anchor_threshold,
            config.max_positive_anchor_num,
            config.max_negative_anchor_ratio
        )

        self.target_layer = TargetLayer(
            config.positive_roi_bbox_threshold,
            config.negative_roi_bbox_threshold,
            config.max_positive_roi_bbox,
            config.max_negative_roi_bbox_ratio
        )

        self.rpn_score_threshold = config.rpn_score_threshold

        if config.roi_align:
            self.roi_ops = [ROIAlign(output_size=config.roi_size, spatial_scale=1 / (2 ** feature_level), sampling_ratio=3) for feature_level in self.backbone_feature_levels]
        else:
            self.roi_ops = [ROIPool(output_size=config.roi_size, spatial_scale=1 / (2 ** feature_level)) for feature_level in self.backbone_feature_levels]


        self.roi_conv_layer = self.backbone.roi_conv_layer
        self.roi_conv_scale = self.backbone.roi_conv_scale

        flatten_roi_dims    = config.roi_channel# * ((config.roi_size[0] + self.roi_conv_scale -1) // self.roi_conv_scale) * ((config.roi_size[1] + self.roi_conv_scale -1) // self.roi_conv_scale)

        self.regression     = torch.nn.Linear(flatten_roi_dims, 4)
        self.classification = torch.nn.Linear(flatten_roi_dims, self.num_classes + 1) # 0 for background

        self.normal_init(self.rpn.conv1, 0, 0.01, False)
        self.normal_init(self.rpn.classification, 0, 0.01, False)
        self.normal_init(self.rpn.regression, 0, 0.01, False)
        self.normal_init(self.classification, 0, 0.01, False)
        self.normal_init(self.regression, 0, 0.001, False)


    def normal_init(self, m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

    def format_rois(self, rpn_bboxes, rpn_scores, batch_idx, image_w, image_h):
        # print('rpn_bboxes.type', rpn_bboxes.type())
        rois   = torch.zeros((0, 5)).type_as(rpn_bboxes)
        # print('rois.type', rois.type())
        e_mask = rpn_scores > self.rpn_score_threshold
        if e_mask.size(0) > 0:
            e_bboxes  = rpn_bboxes[e_mask, :]
            e_bboxes  = self.clip(e_bboxes, image_w, image_h)
            e_scores  = rpn_scores[e_mask]
            e_indices = torch.ones((e_bboxes.size(0), 1)).type_as(e_bboxes) * batch_idx
            rois      = torch.cat([rois, torch.cat([e_indices, e_bboxes], dim=1)], dim=0)
        else:
            e_bboxes = torch.zeros((0, 4)).type_as(rpn_bboxes)
            e_scores = torch.zeros((0)).type_as(rpn_scores)
        return rois, e_bboxes, e_scores

    def clip(self, bboxes, image_w, image_h):
        bboxes[bboxes < 0] = 0
        bboxes[..., 0][bboxes[..., 0] > image_w] = image_w
        bboxes[..., 1][bboxes[..., 1] > image_h] = image_h
        bboxes[..., 2][bboxes[..., 2] > image_w] = image_w
        bboxes[..., 3][bboxes[..., 3] > image_h] = image_h
        return bboxes

    def get_proposal_rois(self, feature_list, rpn_bboxes_list, rpn_scores_list, image_w, image_h):
        batch_rois   = []
        batch_bboxes = []
        batch_scores = []
        for batch_idx in range(feature_list[0].size(0)):
            rois_list    = []
            e_bboxes_list = []
            e_scores_list = []
            for roi_op, feature, rpn_bboxes, rpn_scores in zip(self.roi_ops, feature_list, rpn_bboxes_list, rpn_scores_list):
                one_image_rpn_bboxes = rpn_bboxes[batch_idx, :]
                one_image_rpn_scores = rpn_scores[batch_idx]
                rois, e_bboxes, e_scores = self.format_rois(one_image_rpn_bboxes, one_image_rpn_scores, batch_idx, image_w, image_h)
                rois = roi_op(feature, rois)
                e_bboxes_list.append(e_bboxes)
                e_scores_list.append(e_scores)
                rois_list.append(rois)
            if len(rois_list) > 1:
                one_image_rois   = torch.cat(rois_list, dim=0)
                one_image_bboxes = torch.cat(e_bboxes_list, dim=0)
                one_image_scores = torch.cat(e_scores_list, dim=0)
            else:
                one_image_rois   = rois_list[0]
                one_image_bboxes = e_bboxes_list[0]
                one_image_scores = e_scores_list[0]

            sorted_indices   = torch.argsort(one_image_scores, dim=0, descending=True)
            one_image_rois   = one_image_rois[sorted_indices]
            one_image_bboxes = one_image_bboxes[sorted_indices]
            one_image_scores = one_image_scores[sorted_indices]

            if one_image_rois.size(0) > self.max_outputs_per_image:
                one_image_rois   = one_image_rois[:self.max_outputs_per_image]
                one_image_bboxes = one_image_bboxes[:self.max_outputs_per_image]
                one_image_scores = one_image_scores[:self.max_outputs_per_image]
            elif one_image_rois.size(0) < self.max_outputs_per_image:
                pad_size = self.max_outputs_per_image - one_image_rois.size(0)
                # print('pad_size', pad_size)
                # print('one_image_rois.size()', one_image_rois.size())
                one_image_rois   = F.pad(one_image_rois, pad=[0, 0, 0, 0, 0, 0, 0, pad_size], mode='constant', value=-1)
                # print('one_image_rois.size()', one_image_rois.size())
                # print('one_image_bboxes.size()', one_image_bboxes.size())
                one_image_bboxes = F.pad(one_image_bboxes, pad=[0, 0, 0, pad_size], mode='constant', value=0)
                # print('one_image_bboxes.size()', one_image_bboxes.size())
                one_image_scores = F.pad(one_image_scores, pad=[0, pad_size], mode='constant', value=-1)

            batch_rois.append(one_image_rois)
            batch_bboxes.append(one_image_bboxes)
            batch_scores.append(one_image_scores)
        return torch.stack(batch_rois, dim=0), torch.stack(batch_bboxes, dim=0), torch.stack(batch_scores, dim=0)

    def forward(self, *input):
        if self.training:
            # input should contain gt boxes and labels in training mode
            batch_image, batch_gt_boxes, batch_labels = input
        else:
            batch_image = input[0]
            batch_gt_boxes = batch_labels = None

        # fit image to backbone model and get the features as output
        # with torch.no_grad():
        backbone_features = self.backbone(batch_image)

        image_h = batch_image.size(2)
        image_w = batch_image.size(3)

        feature_list = []
        anchors_list = []

        # get backbone features for each feature level and anchors for each feature
        for level_anchor_sizes, feature_level, feature_idx in zip(self.anchor_sizes, self.use_feature_levels, self.use_feature_idx_list):
            use_feature = backbone_features[feature_idx]
            anchors = get_anchors(anchor_sizes=level_anchor_sizes, anchor_ratios=self.anchor_ratios)
            anchors = anchor_shift(anchors, use_feature.size()[2:4], stride=2 ** feature_level)
            anchors = torch.from_numpy(anchors).float()
            if torch.cuda.is_available():
                anchors = anchors.cuda()
            self.clip(anchors, image_w, image_h)
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

            rois, roi_bboxes, roi_scores = self.get_proposal_rois(feature_list, rpn_bboxes_list, rpn_scores_list, image_w, image_h)

            roi_tags = roi_scores > 0

            # print('rois.size()', rois.size())
            # rois = torch.mean(rois, dim=(3, 4))
            batch_size = rois.size(0)
            roi_number = rois.size(1)

            rois = rois.view(batch_size * roi_number, rois.size(2), rois.size(3), rois.size(4))
            rois = self.roi_conv_layer(rois)
            rois = rois.mean(3).mean(2)
            rois = rois.view(batch_size, roi_number, -1)

            regressions     = self.regression(rois)
            classifications = self.classification(rois)

            # print('rois sum:', torch.sum(rois[0, :4], dim=1))

            regressions_target, classifications_target = self.target_layer(roi_bboxes, roi_tags, batch_gt_boxes, batch_labels)

            regressions_loss     = losses.smooth_l1(regressions, regressions_target, 1)
            classifications_loss = losses.focal_loss(classifications, classifications_target, self.num_classes + 1)

            if len(rpn_bboxes_list) > 1:
                # total loss is the average of losses in each feature level
                rpn_regression_loss     = torch.stack(rpn_regression_loss_list).mean()
                rpn_classification_loss = torch.stack(rpn_classification_loss_list).mean()
            else:
                rpn_regression_loss     = rpn_regression_loss_list[0]
                rpn_classification_loss = rpn_classification_loss_list[0]

            bboxes = _bbox_transform_inv(roi_bboxes, regressions)
            gt_bboxes = _bbox_transform_inv(roi_bboxes, regressions_target)

            mask = regressions_target[..., -1] == 1

            # print('regressions:', regressions[mask][:1])
            # print('regressions_target:', regressions_target[mask][:1])
            # print('roi_bboxes:', roi_bboxes[mask][:1])
            # print('bboxes:', bboxes[mask][:1])
            # print('gt_bboxes:', gt_bboxes[mask][:1])
            # #
            # print('classifications:', classifications[mask][:1])
            # print('classifications_target:', classifications_target[mask][:1])

            bboxes = self.clip(bboxes, image_w, image_h)
            bboxes, scores, labels = get_nms_bboxes(bboxes, classifications, roi_tags, self.max_outputs_per_image, self.rpn_score_threshold)

            return bboxes, scores, labels, rpn_regression_loss, rpn_classification_loss, regressions_loss, classifications_loss
        else:
            rpn_bboxes_list = []
            rpn_scores_list = []
            for feature, anchors in zip(feature_list, anchors_list):
                # use rpn to get proposal bbox for each feature level
                # there is no rpn loss for eval mode
                rpn_bboxes, rpn_scores = self.rpn(feature, anchors, batch_gt_boxes, batch_labels)[:2]
                rpn_bboxes_list.append(rpn_bboxes)
                rpn_scores_list.append(rpn_scores)

            rois, roi_bboxes, roi_scores = self.get_proposal_rois(feature_list, rpn_bboxes_list, rpn_scores_list, image_w, image_h)
            roi_tags = roi_scores > 0

            batch_size = rois.size(0)
            roi_number = rois.size(1)
            rois = rois.view(batch_size * roi_number, rois.size(2), rois.size(3), rois.size(4))
            rois = self.roi_conv_layer(rois)
            # rois.view(batch_size, roi_number, rois.size(1), rois.size(2), rois.size(3))

            rois = rois.view(batch_size, roi_number, -1)
            regressions = self.regression(rois)
            classifications = self.classification(rois)
            bboxes = _bbox_transform_inv(roi_bboxes, regressions)
            bboxes = self.clip(bboxes, image_w, image_h)
            bboxes, scores, labels = get_nms_bboxes(bboxes, classifications, roi_tags, self.max_outputs_per_image, self.rpn_score_threshold)

            return bboxes, scores, labels


if __name__ == '__main__':
    """test Faster Rcnn with only rpn"""
    import os
    import numpy as np
    from .generators.data_process import random_transform_generator, data_aug_func, resize_image_func, image_process_func
    from .generators.coco_generator import CocoGenerator
    from .generators.utils import data_annotations2input_outputs
    from .utils.callbacks import Evaluate
    from torch.utils.data import DataLoader
    import torch.optim.lr_scheduler as lr_scheduler
    from torchvision import transforms
    import logging

    test_config = Config(num_classes=2)

    transform_generator = random_transform_generator(
        rotation_ratio=0.5,
        min_rotation=-0.2,
        max_rotation=0.2,
        translation_ratio=0.5,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        shear_ratio=0.5,
        min_shear=-0.2,
        max_shear=0.2,
        scaling_ratio=0.5,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_ratio=0.5,
        flip_y_ratio=0.,
        prng=None
    )

    train_data_process_func_list = [data_aug_func(transform_generator, 'linear', 'constant', 0.),
                                    resize_image_func((512, 512), 'linear'),
                                    image_process_func(transforms.ToTensor()),
                                    image_process_func(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])),
                                    ]

    valid_data_process_func_list = [resize_image_func((512, 512), 'linear'),
                                    image_process_func(transforms.ToTensor()),
                                    image_process_func(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])),
                                    ]

    train_generator = CocoGenerator(
        data_dir='/mnt/data4/lty/data/coco/train2017',
        annotation_file_path='/mnt/data4/lty/data/coco/annotations/instances_train2017.json',
        data_process_func_list=train_data_process_func_list,
        data_annotations2input_outputs=data_annotations2input_outputs(max_gts=test_config.max_outputs_per_image),
    )
    valid_generator = CocoGenerator(
        data_dir='/mnt/data4/lty/data/coco/val2017',
        annotation_file_path='/mnt/data4/lty/data/coco/annotations/instances_val2017.json',
        data_process_func_list=valid_data_process_func_list,
        data_annotations2input_outputs=data_annotations2input_outputs(max_gts=test_config.max_outputs_per_image),
    )

    print('train_generator num_classes', train_generator.num_classes)

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    DEVICE = torch.device('cuda:0')

    frcnn = FasterRcnn(test_config)

    BATCH_SIZE = 6
    frcnn.to(DEVICE)

    # frcnn.eval()

    print('start run')

    optimizer = torch.optim.Adam(frcnn.parameters(), lr=0.0001)

    train_data_loader = DataLoader(train_generator, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

    # valid_data_loader = DataLoader(valid_generator, batch_size=1, shuffle=False, num_workers=1)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    last_ap = last_loss = 0
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', filename='./%s_train.log' % (test_config.backbone_name), level=logging.INFO)
    for epoch in range(24):
        scheduler.step(epoch)
        frcnn.train()
        for batch_idx, (data, batch_gt_boxes, batch_labels) in enumerate(train_data_loader):
            data, batch_gt_boxes, batch_labels = data.to(DEVICE), batch_gt_boxes.to(DEVICE), batch_labels.to(DEVICE).long()
            # print('data.type()', data.type())
            optimizer.zero_grad()
            bboxes, scores, labels, rpn_regression_loss, rpn_classification_loss, regressions_loss, classifications_loss = frcnn(data, batch_gt_boxes, batch_labels)
            loss = rpn_regression_loss + rpn_classification_loss + classifications_loss + regressions_loss
            loss.backward()

            # print('torch.sum(frcnn.rpn.regression.weight.grad)', torch.sum(frcnn.rpn.regression.weight.grad))
            # print('torch.sum(frcnn.regression.weight.grad)', torch.sum(frcnn.regression.weight.grad))
            # #
            # print('torch.sum(frcnn.rpn.classification.weight.grad)', torch.sum(frcnn.rpn.classification.weight.grad))
            # print('torch.sum(frcnn.classification.weight.grad)', torch.sum(frcnn.classification.weight.grad))
            # print('frcnn.rpn.classification.weight.grad', frcnn.rpn.classification.weight.grad)
            # exit()
            optimizer.step()
            if (batch_idx + 1) % 50 == 0:
                print('epoch{}: {}/{}, loss:{}, rpn reg loss: {}, rpn cls loss: {}, reg loss: {}, cls loss: {}'.format(epoch, (batch_idx + 1) * BATCH_SIZE, len(train_data_loader.dataset),
                                                       loss.item(), rpn_regression_loss.item(), rpn_classification_loss.item(), regressions_loss.item(), classifications_loss.item()))
                average_precisions, test_loss, rpn_reg_loss, rpn_cls_loss, reg_loss, cls_loss = Evaluate(frcnn, valid_generator, device=DEVICE, num_classes=test_config.num_classes, random_valid_samples=20, verbose=False)
                average_precision = float(np.mean([average_precisions[label_idx][0] for label_idx in range(1, test_config.num_classes + 1)]))
                print('average precision: {}, test loss: {}, rpn reg loss: {}, rpn cls loss: {}, reg loss: {}, cls loss:{}'.format(average_precision, test_loss, rpn_reg_loss, rpn_cls_loss, reg_loss, cls_loss))

        average_precisions, test_loss, rpn_reg_loss, rpn_cls_loss, reg_loss, cls_loss = Evaluate(frcnn,
                                                                                                valid_generator,
                                                                                                device=DEVICE,
                                                                                                num_classes=test_config.num_classes,
                                                                                                random_valid_samples=None)
        for label_idx in range(1, test_config.num_classes+1):
            print('class', label_idx, train_generator.class_idx2name(label_idx), 'mAP', average_precisions[label_idx][0])
        average_precision = float(np.mean([average_precisions[label_idx][0] for label_idx in range(1, test_config.num_classes + 1)]))
        print('epoch', epoch, 'average precision: {}, test loss: {}, rpn reg loss: {}, rpn cls loss: {}, reg loss: {}, cls loss:{}'.format(
                average_precision, test_loss, rpn_reg_loss, rpn_cls_loss, reg_loss, cls_loss))
        last_ap = average_precision
        last_loss = test_loss
        logging.info('Epoch#%d: test loss=%.4f, average precision=%.4f' % (epoch, test_loss, average_precision))

    torch.save(frcnn.state_dict(), './%s_ap_%.4f_loss_%.4f.pth' % (test_config.backbone_name, last_ap, last_loss))
        
