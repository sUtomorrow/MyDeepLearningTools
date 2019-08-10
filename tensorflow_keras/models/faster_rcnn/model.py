# -*- coding: utf-8 -*-
# @Time     : 5/23/19 9:34 PM
# @Author   : lty
# @File     : model

import tensorflow as tf
import tensorflow.keras as keras

from .backbones import VggBackbone
from . import layers
from .region_proposal_model import RegionProposalModel

default_anchor_params = {
    'size': [128, 256, 512],
    'ratio': [0.5, 1, 2.0],
}

default_config = {
    'RoiPoolingW': 6,
    'RoiPoolingH': 6,
    'BboxProposalNum': 300,
    'RegionProposalFilters': 512,
    'ClassNum': 2,
}

def FasterRCNNHead(faster_rcnn_model, region_proposal_model):
    '''get predict result of faster-rcnn'''
    regression, classification = faster_rcnn_model.outputs
    rpn_regression, rpn_classification, rpn_proposal_bbox = region_proposal_model.outputs
    boxes = layers.BoundingBox(name='boxes')([rpn_proposal_bbox, regression])
    labels = tf.argmax(classification, dimension=-1, name='labels')
    scores = tf.reduce_max(classification, axis=-1, name='scores')
    return boxes, labels, scores


def FasterRCNN(
        inputs=None,
        inputs_shape=None,
        backbone_name='vgg16',
        anchor_params=default_anchor_params,
        config=default_config,
        backbone_weights=None,
        pretrain_backbone=True,
        rpn_weights=None,
        name='faster_rcnn'
):
    if inputs is None:
        assert (inputs_shape is not None)
        inputs = keras.Input(inputs_shape)

    backbone = VggBackbone(backbone_name, inputs=inputs)

    if pretrain_backbone:
        pretrain_weights = backbone.download_weights()
        backbone.load_weights(pretrain_weights)
    elif backbone_weights:
        backbone.load_weights(backbone_weights)

    backbone_outputs = backbone.get_outputs()
    feature_level = backbone.get_feature_level()

    rpn = RegionProposalModel(
        inputs        = backbone_outputs,
        images        = inputs,
        feature_level = feature_level,
        bbox_num      = config['BboxProposalNum'],
        filter_num    = config['RegionProposalFilters'],
        anchor_params = anchor_params,
        name          = 'rpn',
        weights       = rpn_weights
    )

    rpn_regression, rpn_classification, rpn_proposal_bbox = rpn.outputs

    rpn_proposal_bbox_scale = rpn_proposal_bbox / (2 ** feature_level)

    roi_pooling = layers.RoiPooling(pooling_h=config['RoiPoolingH'], pooling_w=config['RoiPoolingW'])(
        [backbone_outputs, rpn_proposal_bbox_scale])

    roi_pooling_reshape = keras.layers.Flatten()(roi_pooling)

    f = keras.layers.Dense(256, activation='relu')(roi_pooling_reshape)
    f = keras.layers.Dense(256, activation='relu')(f)

    regression = keras.layers.Dense(config['BboxProposalNum'] * 4)(f)
    regression = keras.layers.Reshape((-1, 4), name='regression')(regression)

    classification = keras.layers.Dense(config['BboxProposalNum'] * config['ClassNum'], activation='sigmoid')(f)
    classification = keras.layers.Reshape((-1, config['ClassNum']), name='classification')(classification)

    faster_rcnn_model = keras.models.Model(inputs = [inputs, backbone_outputs, rpn_proposal_bbox], outpus=[regression, classification], name=name)

    return backbone.model, rpn.model, faster_rcnn_model