#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 18:59
# @Author  : Lty
# @File    : region_proposal_model.py

import tensorflow as td
import tensorflow.keras as keras
from . import layers


class RegionProposalModel(object):
    def __init__(self, inputs, images, feature_level, bbox_num, filter_num, anchor_params, name='rpn', weights=None):
        self.inputs = inputs
        self.images = images
        self.feature_level = feature_level
        self.bbox_num = bbox_num
        self.filter_num = filter_num
        self.anchor_params = anchor_params
        self.name = name

        self.outputs = self.build_model()
        self.model = keras.models.Model(inputs=inputs, outputs=self.outputs, name=name)

        if weights:
            self.model.load_weights(weights)

    def build_model(self):
        anchor_num = len(self.anchor_params['size']) * len(self.anchor_params['ratio'])
        prior_anchor = layers.PriorAnchor(self.feature_level, anchor_params=self.anchor_params)(self.inputs)

        feature = keras.layers.Conv2D(self.filter_num, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                      activation='relu')(self.inputs)

        classification = keras.layers.Conv2D(2 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same')(
            feature)

        # [p of background, p of target], shape: [batch_size, anchor_num, 2]
        classification = keras.layers.Reshape((-1, 2))(classification)
        classification = keras.layers.Activation('softmax', axis=-1, name='rpn_classification')(classification)

        regression = keras.layers.Conv2D(4 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same', )(feature)
        regression = keras.layers.Reshape((-1, 4), name='rpn_regression')(regression)  # shape: [batch_size, anchor_num, 4]

        bbox = layers.BoundingBox()([prior_anchor, regression])

        # the coordinates in bbox have same scale with image's origin size
        bbox = layers.BoxClip()([self.images, bbox])

        proposal_bbox = layers.BboxProposal(self.bbox_num, nms_threshold=0.7, name='rpn_proposal_bbox')([classification, bbox])

        return regression, classification, proposal_bbox

