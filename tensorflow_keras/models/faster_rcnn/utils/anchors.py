#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 19:05
# @Author  : Lty
# @File    : anchors.py

import numpy as np

default_anchor_params = {
    'sizes': [256],
    'strides': [32],
    'ratios': [0.5, 1, 2.0],
    'scales': [0.5, 1., 2.0],

}


def iou(b1, b2):
    """
    b1: [N, 4], [x1, y1, x2, y2]
    b2: [M, 4], [x1, y1, x2, y2]
    return iou: [N, M]
    """
    N = b1.shape[0]
    M = b2.shape[0]

    # reshape for broadcast
    b1 = np.tile(np.expand_dims(b1, 1), (1, M, 1))
    b2 = np.tile(np.expand_dims(b2, 0), (N, 1, 1))

    inter_x1 = np.maximum(b1[:, :, 0], b2[:, :, 0])
    inter_x2 = np.minimum(b1[:, :, 2], b2[:, :, 2])

    inter_y1 = np.maximum(b1[:, :, 1], b2[:, :, 1])
    inter_y2 = np.minimum(b1[:, :, 3], b2[:, :, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)

    inter_area = inter_h * inter_w

    w1 = np.maximum((b1[:, :, 2] - b1[:, :, 0]), 0)
    h1 = np.maximum((b1[:, :, 3] - b1[:, :, 1]), 0)

    w2 = np.maximum((b2[:, :, 2] - b2[:, :, 0]), 0)
    h2 = np.maximum((b2[:, :, 3] - b2[:, :, 1]), 0)

    area1 = w1 * h1
    area2 = w2 * h2

    return inter_area / (area1 + area2 - inter_area)


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
    tw = np.log(bw / aw)
    th = np.log(bh / ah)

    return np.stack([tx, ty, tw, th], axis=-1)


def bboxes2outputs(anchors, bboxes, label_idxes, positive_iou=0.5, negative_iou=0.3, class_num=2):
    """
    anchors: [N, 4], [x1, y1, x2, y2]
    bboxes: [M, 4], [x1, y1, x2, y2]
    labels: [M], label idx list
    """

    bboxes = np.array(bboxes, np.float32)
    label_idxes = np.array(label_idxes, np.int32)

    # the last axis is output status: -1 for ignore, 1 for positive, 0 for negative
    regression_outputs = np.zeros(shape=(anchors.shape[0], 4 + 1), dtype=np.float32)
    classification_outputs = np.zeros(shape=(anchors.shape[0], class_num + 1), dtype=np.float32)

    iou_matrix = iou(anchors, bboxes) # [N, M]
    anchors_max_iou = np.max(iou_matrix, axis=-1)
    anchor_max_bbox_indexes = np.argmax(iou_matrix, axis=-1)
    positive_indices = anchors_max_iou >= positive_iou
    ignore_indices = anchors_max_iou >= negative_iou & ~positive_indices
    positive_bbox_indices = anchor_max_bbox_indexes[positive_indices]

    regression_outputs[:, :-1] = bbox_transform(anchors, bboxes[positive_bbox_indices, :])
    regression_outputs[positive_indices, -1] = 1

    classification_outputs[positive_indices, label_idxes[positive_bbox_indices]] = 1

    regression_outputs[ignore_indices, -1] = -1
    classification_outputs[ignore_indices, -1] = -1

    return regression_outputs, classification_outputs


def guess_shapes(image_shape, feature_levles):
    """Guess shapes based on pyramid levels.

    Args
         image_shape: The shape of the image.
         feature_levles: A list of what pyramid levels are used.

    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in feature_levles]
    return image_shapes


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = default_anchor_params['ratios']

    if scales is None:
        scales = default_anchor_params['scales']

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def anchors_for_shape(
    image_shape,
    feature_levles=None,
    anchor_params=None,
    shapes_callback=None,
):
    """ Generators anchors for a given shape.

    Args
        image_shape: The shape of the image.
        feature_levles: List of ints representing which pyramids to use (defaults to [2, 3, 4, 5, 6]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.

    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if feature_levles is None:
        feature_levles = [2, 3, 4, 5, 6]

    if anchor_params is None:
        anchor_params = default_anchor_params

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, feature_levles)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(feature_levles):
        anchors = generate_anchors(
            base_size=anchor_params['sizes'][idx],
            ratios=anchor_params['ratios'],
            scales=anchor_params['scales']
        )
        shifted_anchors = shift(image_shapes[idx], anchor_params['strides'][idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def rpn_annotations2outputs(image_shape, anchor_params, feature_levels, annotations, positive_iou=0.5, negative_iou=0.3, class_num=2):
    """"""
    bboxes = annotations['bboxes']
    # all annotations is foreground class, class idx = 0, the rpn output only one class probability
    label_idxes = np.zeros(annotations['label_idxes'].shape, dtype=np.int32)

    anchors = anchors_for_shape(image_shape, feature_levels, anchor_params, guess_shapes)

    outputs = bboxes2outputs(anchors, bboxes, label_idxes, positive_iou, negative_iou, class_num)

    return outputs


def rpn_groups_annotations2outputs(anchor_params, feature_levels, positive_iou=0.5, negative_iou=0.3, class_num=2):
    def _rpn_groups_annotations2outputs(group_datas, group_annotations):
        image_shape = group_datas.shape[1:]
        group_outputs = []
        for annotations in group_annotations:
            group_outputs.append(rpn_annotations2outputs(image_shape, anchor_params, feature_levels, annotations, positive_iou, negative_iou, class_num))
        return np.array(group_outputs, dtype=np.float32)

    return _rpn_groups_annotations2outputs


def group_annotations2outputs_from_proposal_bboxes(positive_iou=0.5, negative_iou=0.3, class_num=80):
    def _group_annotations2outputs_from_proposal_bboxes(group_proposal_bboxes, group_annotations):
        group_outputs = []
        for annotations in group_annotations:
            bboxes = annotations['bboxes']
            label_idxes = annotations['label_idxes']
            group_outputs.append(bboxes2outputs(group_proposal_bboxes, bboxes, label_idxes, positive_iou, negative_iou, class_num))
        return np.array(group_outputs, dtype=np.float32)

    return _group_annotations2outputs_from_proposal_bboxes