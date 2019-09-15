# -*- coding: utf-8 -*-
# @Time     : 5/23/19 9:34 PM
# @Author   : lty
# @File     : model

import numpy as np
import tensorflow as tf
import keras
from tensorflow_keras.backbones import VggBackbone
import layers
from utils.losses import focal, smooth_l1

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
        weights=None,
        name='rpn'
):
    image_inputs = keras.layers.Input(shape=image_inputs_shape, name='rpn_image_inputs')
    feature_inputs = keras.layers.Input(shape=feature_inputs_shape, name='rpn_feature_inputs')

    anchor_num = len(anchor_params['scales']) * len(anchor_params['ratios'])

    prior_anchor = layers.PriorAnchor(0, feature_level, anchor_params=anchor_params)(feature_inputs)

    feature = keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(feature_inputs)

    classifications = keras.layers.Conv2D(anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same')(feature)

    # [p of background, p of target], shape: [batch_size, anchor_num, 1]
    classifications = keras.layers.Reshape((-1, 1))(classifications)
    classifications = keras.layers.Activation(activation='sigmoid', name='rpn_classification')(classifications)

    regressions = keras.layers.Conv2D(4 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same')(feature)
    regressions = keras.layers.Reshape((-1, 4), name='rpn_regression')(regressions)  # shape: [batch_size, anchor_num, 4]

    bboxes = layers.BoundingBox()([prior_anchor, regressions])

    # the coordinates in bbox have same scale with image's origin size
    bboxes = layers.BoxClip()([image_inputs, bboxes])

    proposal_bboxes = layers.BboxProposal(bbox_num, nms_threshold=0.7, name='rpn_proposal_bbox')([classifications, bboxes])

    model = keras.models.Model(inputs=[image_inputs, feature_inputs], outputs=[regressions, classifications, proposal_bboxes, prior_anchor], name=name)

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


def RCNNHeadModel(feature_inputs_shape, proposal_bbox_inputs_shape, feature_level, model_params, weights=None, name='rcnn_head'):

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

    model = keras.models.Model(inputs=[feature_inputs, proposal_bbox_inputs], outputs=[regressions, classifications], name=name)

    if weights:
        model.load_weights(weights)

    return model


def FasterRCNNComponents(
        anchor_params,
        model_params,
        imagenet_backbone=True,
        backbone_weights=None,
        rpn_weights=None,
        rcnn_weights=None
):
    backbone = VggBackbone(model_params['BackboneName'], inputs=None, inputs_shape=model_params['ImageInputShape'], include_top=False, name='backbone')

    if imagenet_backbone:
        backbone_weights = backbone.download_weights()
    if backbone_weights is not None:
        backbone.load_weights(backbone_weights)

    # only use the last feature map
    backbone_outputs = backbone.model.outputs[-1]
    feature_level = backbone.feature_levels[-1]

    rpn_model = RegionProposalModel(
        image_inputs_shape   = model_params['ImageInputShape'],
        feature_inputs_shape = backbone_outputs.shape.as_list()[1:],
        feature_level = feature_level,
        bbox_num      = model_params['BboxProposalNum'],
        filter_num    = model_params['RegionProposalFilters'],
        anchor_params = anchor_params,
        weights       = rpn_weights,
    )

    rcnn_model = RCNNHeadModel(
        feature_inputs_shape       = backbone_outputs.shape.as_list()[1:],
        proposal_bbox_inputs_shape = rpn_model.outputs[2].shape.as_list()[1:],
        feature_level              = feature_level,
        model_params               = model_params,
        weights                    = rcnn_weights
    )

    return backbone, rpn_model, rcnn_model


def FasterRcnn(config, train=False, imagenet_backbone=False, backbone_weights=None, rpn_weights=None, rcnn_weights=None):
    backbone, rpn_model, rcnn_model = FasterRCNNComponents(
        config['anchor_params'],
        config['model_params'],
        imagenet_backbone=imagenet_backbone,
        backbone_weights=backbone_weights,
        rpn_weights=rpn_weights,
        rcnn_weights=rcnn_weights,
    )

    backbone_model = backbone.model
    return_list = [backbone.image_preprocess_func, backbone_model, rpn_model, rcnn_model]

    multi_gpu = len(config['gpu'].split(','))
    if multi_gpu > 1:
        from keras.utils import multi_gpu_model
        backbone_model = multi_gpu_model(backbone_model, gpus=multi_gpu)
        rpn_model = multi_gpu_model(rpn_model, gpus=multi_gpu)
        rcnn_model = multi_gpu_model(rcnn_model, gpus=multi_gpu)

    print('backbone:')
    backbone_model.summary()
    print('rpn_model:')
    rpn_model.summary()
    print('rcnn_model:')
    rcnn_model.summary()

    image_inputs = keras.layers.Input(shape=config['model_params']['ImageInputShape'], name='image_inputs')

    backbone_outputs = backbone_model(image_inputs)
    rpn_regressions, rpn_classifications, rpn_proposal_bboxes, prior_anchor = rpn_model([image_inputs, backbone_outputs[-1]])
    frcnn_regressions, frcnn_classifications = rcnn_model([backbone_outputs[-1], rpn_proposal_bboxes])

    faster_rcnn_inference = FasterRCNNInferenceModel(image_inputs, rpn_proposal_bboxes, frcnn_regressions, frcnn_classifications)
    print('faster_rcnn_inference:')
    faster_rcnn_inference.summary()

    return_list.append(faster_rcnn_inference)

    if train:
        gt_boxes_inputs = keras.layers.Input(shape=(None, 5), name='gt_boxes_inputs')
        gt_class_idxes_inputs = keras.layers.Input(shape=(None, 2), name='gt_class_idxes_inputs')

        rpn_regression_targets, rpn_classification_targets = layers.RpnTarget(
            config['model_params']['RpnPositiveIou'],
            config['model_params']['RpnNegativeIou']
        )([prior_anchor, gt_boxes_inputs, gt_class_idxes_inputs])

        frcnn_regression_targets, frcnn_classification_targets = layers.FrcnnTarget(
            config['model_params']['FrcnnPositiveIou'],
            config['model_params']['FrcnnNegativeIou'],
            config['model_params']['ClassNum']
        )([rpn_proposal_bboxes, gt_boxes_inputs, gt_class_idxes_inputs])

        rpn_cls_loss_func = focal(0.25, 2.0)
        rpn_reg_loss_func = smooth_l1()

        frcnn_cls_loss_func = focal(0.25, 2.0)
        frcnn_reg_loss_func = smooth_l1()

        # 定义rpn损失layer
        rpn_reg_loss = keras.layers.Lambda(lambda x: rpn_reg_loss_func(*x), name='rpn_reg_loss')(
            [rpn_regression_targets, rpn_regressions])

        rpn_cls_loss = keras.layers.Lambda(lambda x: rpn_cls_loss_func(*x), name='rpn_cls_loss')(
            [rpn_classification_targets, rpn_classifications])

        frcnn_reg_loss = keras.layers.Lambda(lambda x: frcnn_reg_loss_func(*x), name='frcnn_reg_loss')(
            [frcnn_regression_targets, frcnn_regressions])

        frcnn_cls_loss = keras.layers.Lambda(lambda x: frcnn_cls_loss_func(*x), name='frcnn_cls_loss')(
            [frcnn_classification_targets, frcnn_classifications])

        faster_rcnn_training = keras.models.Model(
            inputs=[image_inputs, gt_boxes_inputs, gt_class_idxes_inputs],# + rpn_model.inputs + rcnn_model.inputs,
            outputs=[rpn_reg_loss, rpn_cls_loss, frcnn_reg_loss, frcnn_cls_loss]
        )
        print('faster_rcnn_training:')
        faster_rcnn_training.summary()

        return_list.append(faster_rcnn_training)
    return return_list




