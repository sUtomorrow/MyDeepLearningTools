# -*- coding: utf-8 -*-
# @Time     : 5/23/19 9:34 PM
# @Author   : lty
# @File     : model

import numpy as np
import tensorflow as tf
import keras
from tensorflow_keras.backbones import VggBackbone
import layers


default_anchor_params = {
    'size': [128, 256, 512],
    'ratio': [0.5, 1, 2.0],
}


default_config = {
    'RoiPoolingW': 6,
    'RoiPoolingH': 6,
    'BboxProposalNum': 300, # image的大小和anchor params的设置，必须要产生多于BboxProposalNum的anchor个数
    'RegionProposalFilters': 512,
    'ClassNum': 2,
}


def RegionProposalModel(
        inputs,
        images,
        feature_level,
        bbox_num,
        filter_num,
        anchor_params,
        name='rpn',
        weights=None
):
    anchor_num = len(anchor_params['size']) * len(anchor_params['ratio'])
    prior_anchor = layers.PriorAnchor(feature_level, anchor_params=anchor_params)(inputs)

    feature = keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)

    classifications = keras.layers.Conv2D(2 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same')(feature)

    # [p of background, p of target], shape: [batch_size, anchor_num, 2]
    classifications = keras.layers.Reshape((-1, 2))(classifications)
    classifications = keras.layers.Softmax(axis=-1, name='rpn_classification')(classifications)

    regressions = keras.layers.Conv2D(4 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same')(feature)
    regressions = keras.layers.Reshape((-1, 4), name='rpn_regression')(regressions)  # shape: [batch_size, anchor_num, 4]

    bboxes = layers.BoundingBox()([prior_anchor, regressions])

    # the coordinates in bbox have same scale with image's origin size
    bboxes = layers.BoxClip()([images, bboxes])

    proposal_bboxes = layers.BboxProposal(bbox_num, nms_threshold=0.7, name='rpn_proposal_bbox')([classifications, bboxes])

    model = keras.models.Model(inputs=images, outputs=[regressions, classifications, proposal_bboxes], name=name)

    if weights:
        model.load_weights(weights)

    return model



def FasterRCNNHead(faster_rcnn_model, region_proposal_model):
    '''get predict result of faster-rcnn'''
    regressions, classifications = faster_rcnn_model.outputs
    rpn_regressions, rpn_classifications, rpn_proposal_bboxes = region_proposal_model.outputs
    boxes = layers.BoundingBox(name='boxes')([rpn_proposal_bboxes, regressions])
    labels = layers.Label(name='labels')(classifications)
    scores = layers.Score(name='scores')(classifications)
    model = keras.models.Model(inputs=faster_rcnn_model.input, outputs=[boxes, labels, scores])
    return model


def FasterRCNN(
        inputs,
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
        inputs = keras.layers.Input(inputs_shape)

    backbone = VggBackbone(backbone_name, inputs=inputs, include_top=False)

    if pretrain_backbone:
        if backbone_weights:
            backbone.load_weights(backbone_weights)
        else:
            pretrain_weights = backbone.download_weights()
            backbone.load_weights(pretrain_weights)

    # only use the last feature map
    backbone_outputs = backbone.outputs[-1]
    feature_level = backbone.feature_levels[-1]

    rpn_model = RegionProposalModel(
        inputs        = backbone_outputs,
        images        = inputs,
        feature_level = feature_level,
        bbox_num      = config['BboxProposalNum'],
        filter_num    = config['RegionProposalFilters'],
        anchor_params = anchor_params,
        name          = 'rpn',
        weights       = rpn_weights
    )

    rpn_regression, rpn_classification, rpn_proposal_bbox = rpn_model.outputs

    scale = 1 / (2 ** feature_level)
    rpn_proposal_bbox_scale = layers.Rescale(scale=scale)(rpn_proposal_bbox)

    roi_pooling = layers.RoiPooling(pooling_h=config['RoiPoolingH'], pooling_w=config['RoiPoolingW'])(
        [backbone_outputs, rpn_proposal_bbox_scale]) # roi_pooling: (batch_size, roi_number, pooling_h, pooling_w, channels)

    # use 3D Convolution instead locally Dense layer
    f = keras.layers.Conv3D(256, kernel_size=(1, config['RoiPoolingH'], config['RoiPoolingW']), padding='valid', activation='relu')(roi_pooling)
    f = keras.layers.Conv3D(256, kernel_size=(1, 1, 1), padding='valid', activation='relu')(f)

    regressions = keras.layers.Conv3D(4, kernel_size=(1, 1, 1), padding='valid')(f)
    regressions = keras.layers.Reshape((-1, 4), name='regression')(regressions)

    classifications = keras.layers.Conv3D(config['ClassNum'], kernel_size=(1, 1, 1), activation='sigmoid')(f)
    classifications = keras.layers.Reshape((-1, config['ClassNum']), name='classification')(classifications)

    faster_rcnn_model = keras.models.Model(inputs=inputs, outputs=[regressions, classifications], name=name)

    return backbone.model, rpn_model, faster_rcnn_model

if __name__ == '__main__':
    # test faster-rcnn
    backbone_model, rpn_model, faster_rcnn_model = FasterRCNN(
        inputs=None,
        inputs_shape=(256, 256, 3),
        backbone_name='vgg16',
        anchor_params=default_anchor_params,
        config=default_config,
        backbone_weights=None,
        pretrain_backbone=True,
        rpn_weights=None,
        name='faster_rcnn'
    )

    training_faster_rcnn_model = faster_rcnn_model
    inference_faster_rcnn_model = FasterRCNNHead(faster_rcnn_model, rpn_model)

    print(faster_rcnn_model.summary())

    print(inference_faster_rcnn_model.summary())

    print(inference_faster_rcnn_model.output)



