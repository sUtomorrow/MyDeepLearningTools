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
    'sizes': [256],
    'strides': [32],
    'ratios': [0.5, 1, 2.0],
    'scales': [0.5, 1.0, 2.0]
}


default_config = {
    'RoiPoolingW': 6,
    'RoiPoolingH': 6,
    'BboxProposalNum': 300, # image的大小和anchor params的设置，必须要产生多于BboxProposalNum的anchor个数
    'RegionProposalFilters': 512,
    'ClassNum': 2,
}


def RegionProposalModel(
        image_inputs_shape,
        feature_inputs_shape,
        feature_level,
        bbox_num,
        filter_num,
        anchor_params,
        name='rpn',
        weights=None
):
    image_inputs = keras.layers.Input(shape=image_inputs_shape, name='rpn_image_inputs')
    feature_inputs = keras.layers.Input(shape=feature_inputs_shape, name='rpn_feature_inputs')

    anchor_num = len(anchor_params['scales']) * len(anchor_params['ratios'])

    prior_anchor = layers.PriorAnchor(0, feature_level, anchor_params=anchor_params)(feature_inputs)

    feature = keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(feature_inputs)

    classifications = keras.layers.Conv2D(2 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same')(feature)

    # [p of background, p of target], shape: [batch_size, anchor_num, 2]
    classifications = keras.layers.Reshape((-1, 2))(classifications)
    classifications = keras.layers.Softmax(axis=-1, name='rpn_classification')(classifications)

    regressions = keras.layers.Conv2D(4 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same')(feature)
    regressions = keras.layers.Reshape((-1, 4), name='rpn_regression')(regressions)  # shape: [batch_size, anchor_num, 4]

    bboxes = layers.BoundingBox()([prior_anchor, regressions])

    # the coordinates in bbox have same scale with image's origin size
    bboxes = layers.BoxClip()([image_inputs, bboxes])

    proposal_bboxes = layers.BboxProposal(bbox_num, nms_threshold=0.7, name='rpn_proposal_bbox')([classifications, bboxes])

    model = keras.models.Model(inputs=[image_inputs, feature_inputs], outputs=[regressions, classifications, proposal_bboxes], name=name)

    if weights:
        model.load_weights(weights)

    return model


def FasterRCNNInferenceModel(image_inputs, rpn_proposal_bboxes, faster_rcnn_regressions, faster_rcnn_classifications, name='faster_rcnn_inference'):
    '''get predict result of faster-rcnn'''
    boxes  = layers.BoundingBox()([rpn_proposal_bboxes, faster_rcnn_regressions])
    boxes  = layers.BoxClip(name='boxes')([image_inputs, boxes])
    labels = layers.Label(name='labels')(faster_rcnn_classifications)
    scores = layers.Score(name='scores')(faster_rcnn_classifications)
    return keras.models.Model(inputs=image_inputs, outputs=[boxes, labels, scores], name=name)


def FasterRCNNModel(feature_inputs_shape, proposal_bbox_inputs_shape, feature_level, model_params, name='faster_rcnn'):

    feature_inputs       = keras.layers.Input(shape=feature_inputs_shape, name='feature_inputs')
    proposal_bbox_inputs = keras.layers.Input(shape=proposal_bbox_inputs_shape, name='proposal_bbox_inputs')

    scale = 1 / (2 ** feature_level)
    proposal_bbox_scale = layers.Rescale(scale=scale)(proposal_bbox_inputs)

    roi_pooling = layers.RoiPooling(pooling_h=model_params['RoiPoolingH'], pooling_w=model_params['RoiPoolingW'])(
        [feature_inputs, proposal_bbox_scale])  # roi_pooling: (batch_size, roi_number, pooling_h, pooling_w, channels)

    # use 3D Convolution instead locally Dense layer
    f = keras.layers.Conv3D(256, kernel_size=(1, model_params['RoiPoolingH'], model_params['RoiPoolingW']),
                            padding='valid', activation='relu')(roi_pooling)
    f = keras.layers.Conv3D(256, kernel_size=(1, 1, 1), padding='valid', activation='relu')(f)

    regressions = keras.layers.Conv3D(4, kernel_size=(1, 1, 1), padding='valid')(f)
    regressions = keras.layers.Reshape((-1, 4), name='regression')(regressions)

    classifications = keras.layers.Conv3D(model_params['ClassNum'], kernel_size=(1, 1, 1), activation='sigmoid')(f)
    classifications = keras.layers.Reshape((-1, model_params['ClassNum']), name='classification')(classifications)

    return keras.models.Model(inputs=[feature_inputs, proposal_bbox_inputs], outputs=[regressions, classifications], name=name)


def FasterRCNNComponents(
        anchor_params,
        model_params,
        image_shape=(512, 512, 3),
        proposal_bbox_shape=(None, 4),
        backbone_name='vgg16',
        backbone_weights=None,
        pretrain_backbone=True,
        rpn_weights=None,
        name='faster_rcnn'
):
    backbone = VggBackbone(backbone_name, inputs=None, inputs_shape=image_shape, include_top=False, name=backbone_name)

    if pretrain_backbone:
        if backbone_weights:
            backbone.load_weights(backbone_weights)
        else:
            pretrain_weights = backbone.download_weights()
            backbone.load_weights(pretrain_weights)

    # only use the last feature map
    backbone_outputs = backbone.model.outputs[-1]
    feature_level = backbone.feature_levels[-1]

    rpn_model = RegionProposalModel(
        image_inputs_shape   = image_shape,
        feature_inputs_shape = backbone_outputs.shape.as_list()[1:],
        feature_level = feature_level,
        bbox_num      = model_params['BboxProposalNum'],
        filter_num    = model_params['RegionProposalFilters'],
        anchor_params = anchor_params,
        name          = 'rpn',
        weights       = rpn_weights
    )

    faster_rcnn_model = FasterRCNNModel(
        feature_inputs_shape       = backbone_outputs.shape.as_list()[1:],
        proposal_bbox_inputs_shape = proposal_bbox_shape,
        feature_level              = feature_level,
        model_params               = model_params,
        name                       = name,
    )
    return backbone.model, rpn_model, faster_rcnn_model


if __name__ == '__main__':

    input_shape = (512, 512, 3)
    # train region proposal model and faster-rcnn model split
    image_inputs = keras.layers.Input(shape=input_shape, name='image')
    proposal_bbox_inputs = keras.layers.Input(shape=(None, 4), name='proposal_bbox')

    backbone_model, rpn_model, faster_rcnn_model = FasterRCNNComponents(
        anchor_params=default_anchor_params,
        model_params=default_config,
        image_shape=input_shape,
        proposal_bbox_shape=(None, 4),
        backbone_name='vgg16',
        backbone_weights=None,
        pretrain_backbone=False,
        rpn_weights=None,
        name='faster_rcnn'
    )

    backbone_outputs = backbone_model(image_inputs)[-1]

    # print('backbone_outputs.shape', backbone_outputs.shape)
    # print('backbone_outputs._keras_shape', backbone_outputs._keras_shape)

    rpn_regressions, rpn_classifications, rpn_proposal_bboxes = rpn_model([image_inputs, backbone_outputs])

    faster_rcnn_regressions_from_input_box, faster_rcnn_classifications_from_input_box = faster_rcnn_model([backbone_outputs, proposal_bbox_inputs])

    faster_rcnn_regressions, faster_rcnn_classifications = faster_rcnn_model([backbone_outputs, rpn_proposal_bboxes])

    rpn_model_training = keras.models.Model(
        inputs=image_inputs,
        outputs=[rpn_regressions, rpn_classifications],
        name='rpn_model_training'
    )

    frcnn_model_training_1 = keras.models.Model(
        inputs=[image_inputs, proposal_bbox_inputs],
        outputs=[faster_rcnn_regressions_from_input_box, faster_rcnn_classifications_from_input_box],
        name='frcnn_model_training_1'
    )

    frcnn_model_training_2 = keras.models.Model(
        inputs=image_inputs,
        outputs=[faster_rcnn_regressions, faster_rcnn_classifications],
        name='frcnn_model_training_2'
    )

    frcnn_model_inference = FasterRCNNInferenceModel(
        image_inputs=image_inputs,
        rpn_proposal_bboxes = rpn_proposal_bboxes,
        faster_rcnn_regressions=faster_rcnn_regressions,
        faster_rcnn_classifications=faster_rcnn_classifications,
        name='frcnn_model_inference'
    )

    print(rpn_model_training.summary())
    print(frcnn_model_training_1.summary())
    print(frcnn_model_training_2.summary())
    print(frcnn_model_inference.summary())



