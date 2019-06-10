#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 18:59
# @Author  : Lty
# @File    : region_proposal_model.py

import tensorflow as td
import tensorflow.keras as keras
from . import layers


class RegionProposalModel(object):
    def __init__(self, inputs, images, feature_level, bbox_num, filter_num, anchor_params, name='rpn'):
        inputs = inputs
        anchor_num = len(anchor_params['size']) * len(anchor_params['ratio'])
        prior_anchor = layers.PriorAnchor(feature_level, anchor_params=anchor_params)(inputs)

        feature = keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                      activation='relu')(inputs)

        classification = keras.layers.Conv2D(2 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same')(
            feature)

        # [p of background, p of target], shape: [batch_size, anchor_num, 2]
        classification = keras.layers.Reshape((-1, 2))(classification)
        classification = keras.layers.Activation('softmax', axis=-1)(classification)

        regression = keras.layers.Conv2D(4 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same', )(feature)
        regression = keras.layers.Reshape((-1, 4))(regression) # shape: [batch_size, anchor_num, 4]

        bbox = layers.BoundingBox()([prior_anchor, regression])

        # the coordinates in bbox have same scale with image's origin size
        bbox = layers.BoxClip()([images, bbox])

        proposal_bbox = layers.BboxProposal(bbox_num, nms_threshold=0.7)([classification, bbox])

        RPN = keras.models.Model(inputs=inputs, outputs=[regression, classification, proposal_bbox], name=name)
