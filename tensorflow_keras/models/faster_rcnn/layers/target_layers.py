# -*- coding: utf-8 -*-
# @Time     : 8/20/19 10:27 AM
# @Author   : lty
# @File     : target_layers

import tensorflow as tf
import keras

def remove_pad(input_tensor):
    """remove padding, there should be a status column in input_tensor
    """
    pad_tag = input_tensor[..., -1]
    real_size = tf.cast(tf.reduce_sum(pad_tag), tf.int32)
    return input_tensor[:real_size, :-1]


def bbox_transform(anchors, bboxes):
    """ prior anchor should be [x1, y1, x2, y2],
        bbox should be [bx1, by1, bx2, by2]

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

    bx = (bboxes[:, 0] + bboxes[:, 2]) / 2
    by = (bboxes[:, 1] + bboxes[:, 3]) / 2
    bw = bboxes[:, 2] - bboxes[:, 0]
    bh = bboxes[:, 3] - bboxes[:, 1]

    tx = (bx - ax) / aw
    ty = (by - ay) / ah
    tw = tf.log(bw / aw)
    th = tf.log(bh / ah)

    return tf.stack([tx, ty, tw, th], axis=-1)


def iou(b1, b2):
    """
    b1: [N, 4], [x1, y1, x2, y2]
    b2: [M, 4], [x1, y1, x2, y2]
    return iou: [N, M]
    """
    N = tf.shape(b1)[0]
    M = tf.shape(b2)[0]

    # reshape for broadcast
    b1 = tf.tile(tf.expand_dims(b1, 1), (1, M, 1))
    b2 = tf.tile(tf.expand_dims(b2, 0), (N, 1, 1))

    inter_x1 = tf.maximum(b1[:, :, 0], b2[:, :, 0])
    inter_x2 = tf.minimum(b1[:, :, 2], b2[:, :, 2])

    inter_y1 = tf.maximum(b1[:, :, 1], b2[:, :, 1])
    inter_y2 = tf.minimum(b1[:, :, 3], b2[:, :, 3])

    inter_w = tf.maximum(inter_x2 - inter_x1, 0)
    inter_h = tf.maximum(inter_y2 - inter_y1, 0)

    inter_area = inter_h * inter_w

    w1 = tf.maximum((b1[:, :, 2] - b1[:, :, 0]), 0)
    h1 = tf.maximum((b1[:, :, 3] - b1[:, :, 1]), 0)

    w2 = tf.maximum((b2[:, :, 2] - b2[:, :, 0]), 0)
    h2 = tf.maximum((b2[:, :, 3] - b2[:, :, 1]), 0)

    area1 = w1 * h1
    area2 = w2 * h2

    return inter_area / (area1 + area2 - inter_area)


def bboxes2targets(positive_iou=0.5, negative_iou=0.3, class_num=2):

    def _bboxes2targets(inputs):

        """
        prior_anchors: [N, 4], [x1, y1, x2, y2]
        bboxes: [M, 5], [x1, y1, x2, y2, padding_status]
        labels: [M, 2], [label idx, padding_status]
        """
        prior_anchors, gt_bboxes, gt_class_idxes = inputs
        print(prior_anchors.shape)
        print(gt_bboxes.shape)
        print(gt_class_idxes.shape)
        gt_bboxes = remove_pad(gt_bboxes)
        gt_class_idxes = remove_pad(gt_class_idxes)[:, 0]

        iou_matrix = iou(prior_anchors, gt_bboxes) # [N, M]

        print('iou_matrix.shape', iou_matrix.shape)

        anchors_max_iou = tf.reduce_max(iou_matrix, axis=-1)
        anchor_max_bbox_indexes = tf.argmax(iou_matrix, axis=-1)

        positive_status = anchors_max_iou >= positive_iou
        ignore_status = (anchors_max_iou >= negative_iou) & ~positive_status

        positive_indices = tf.where(positive_status)
        ignore_indices = tf.where(ignore_status)

        print('positive_indices.shape', positive_indices.shape)
        print('ignore_indices.shape', ignore_indices.shape)

        # positive_bbox_indices = tf.gather(anchor_max_bbox_indexes, positive_indices)
        #
        # print('positive_bbox_indices.shape', positive_bbox_indices.shape)

        regression_targets = bbox_transform(prior_anchors, tf.gather(gt_bboxes, anchor_max_bbox_indexes))
        targets_status = tf.expand_dims(tf.where(
            positive_status,
            tf.ones(tf.shape(prior_anchors)[0]),
            tf.where(
                ignore_status,
                tf.ones(tf.shape(prior_anchors)[0]) * -1,
                tf.zeros(tf.shape(prior_anchors)[0])
            )), axis=-1)

        classification_targets = tf.one_hot(tf.cast(tf.gather(gt_class_idxes, anchor_max_bbox_indexes), 'int32'), depth=class_num)
        # classification_targets[positive_indices, gt_class_idxes[positive_bbox_indices]] = 1

        regression_targets = tf.concat([regression_targets, targets_status], axis=-1)
        classification_targets = tf.concat([classification_targets, targets_status], axis=-1)

        return (regression_targets, classification_targets)

    return _bboxes2targets


class RpnTarget(keras.layers.Layer):
    def __init__(self, positive_iou, negative_iou, **kwargs):
        self.positive_iou = positive_iou
        self.negative_iou = negative_iou
        self.bboxes2targets = bboxes2targets(self.positive_iou, self.negative_iou, 1)
        super(RpnTarget, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
            prior_anchors: [B, N, 4], [x1, y1, x2, y2]
            gt_bboxes: [B, M, 5], [x1, y1, x2, y2]
            gt_class_idxes: [B, 2], label idx
        """
        prior_anchors, gt_bboxes, gt_class_idxes = inputs

        gt_class_idxes = tf.zeros_like(gt_class_idxes) # all gt is foreground, class index is 0

        regression_targets, classification_targets = tf.map_fn(self.bboxes2targets, (prior_anchors, gt_bboxes, gt_class_idxes), dtype=(keras.backend.floatx(), keras.backend.floatx()))

        return [regression_targets, classification_targets]

    def compute_output_shape(self, input_shape):
        anchors_shape = input_shape[0]
        return [anchors_shape[:-1] + (5,), anchors_shape[:-1] + (2,)]


class FrcnnTarget(keras.layers.Layer):
    def __init__(self, positive_iou, negative_iou, class_num, **kwargs):
        self.positive_iou = positive_iou
        self.negative_iou = negative_iou
        self.class_num = class_num
        self.bboxes2targets = bboxes2targets(self.positive_iou, self.negative_iou, self.class_num)
        super(FrcnnTarget, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
            proposal_bboxes: [B, N, 4], [x1, y1, x2, y2]
            gt_bboxes: [B, M, 5], [x1, y1, x2, y2]
            gt_class_idxes: [B, 2], label idx
        """
        proposal_bboxes, gt_bboxes, gt_class_idxes = inputs

        regression_targets, classification_targets = tf.map_fn(self.bboxes2targets, (proposal_bboxes, gt_bboxes, gt_class_idxes), dtype=(keras.backend.floatx(), keras.backend.floatx()))

        return [regression_targets, classification_targets]

    def compute_output_shape(self, input_shape):
        proposal_bboxes_shape = input_shape[0]
        return [proposal_bboxes_shape[:-1] + (5,), proposal_bboxes_shape[:-1] + (self.class_num+1,)]
