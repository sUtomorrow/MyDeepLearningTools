# -*- coding: utf-8 -*-
# @Time     : 5/23/19 9:34 PM
# @Author   : lty
# @File     : model


import tensorflow.keras as keras

from .backbones import VggBackbone
from . import layers

default_anchor_params = {
    'size': [128, 256, 512],
    'ratio': [0.5, 1, 2.0],
}

default_config = {
    'RoiPoolingW': 6,
    'RoiPoolingH': 6,
}


def RegionProposalNet(inputs, prior_anchor, filter_num, anchor_num):

    inputs = keras.layers.Conv2D(filter_num, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)

    classification = keras.layers.Conv2D(2 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    classification = keras.layers.Reshape((-1, anchor_num, 2))(classification)
    classification = keras.layers.Activation('softmax', axis=-1)(classification)

    regression = keras.layers.Conv2D(4 * anchor_num, kernel_size=(1, 1), strides=(1, 1), padding='same', )(inputs)
    regression = keras.layers.Reshape((-1, anchor_num, 4))(regression)



def FasterRCNN(inputs=None, inputs_shape=None, backbone_name='vgg16', anchor_params=default_anchor_params, config=default_config):
    if inputs is None:
        assert(inputs_shape is not None)
        inputs = keras.Input(inputs_shape)

    backbone = VggBackbone(backbone_name, inputs=inputs)

    backbone_outputs = backbone.get_outputs()
    feature_level = backbone.get_feature_level()

    prior_anchor = layers.PriorAnchor(feature_level, anchor_params=anchor_params)(backbone_outputs)

    roi_pooling = layers.RoiPooling(pooling_h=config['RoiPoolingH'], pooling_w=config['RoiPoolingW'])([backbone_outputs, prior_anchor])







