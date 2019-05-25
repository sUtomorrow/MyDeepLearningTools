# -*- coding: utf-8 -*-
# @Time     : 5/25/19 10:40 AM
# @Author   : lty
# @File     : vgg

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.keras.applications import vgg16, vgg19
from .backbone import Backbone

VGG16_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
VGG19_WEIGHTS = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

class VggBackbone(Backbone):
    def __init__(self, backbone_name, inputs=None, inputs_shape=None):
        self.backbone_name = backbone_name
        if inputs is None:
            self.inputs = keras.Input(shape=inputs_shape)
        else:
            self.inputs = inputs
        super(VggBackbone, self).__init__()

    def build_network(self):
        if self.backbone_name == 'vgg16':
            self.model = vgg16.VGG16(False, None, input_tensor=self.inputs)
            self.outputs = self.model.get_layer('block5_pool').output()
            self.feature_level = 5
        elif self.backbone_name == 'vgg19':
            self.model = vgg19.VGG19(False, None, input_tensor=self.inputs)
            self.outputs = self.model.get_layer('block5_pool').output()
            self.feature_level = 5
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(self.backbone_name))

    def download_weights(self):
        weights_path = None

        if self.backbone_name == 'vgg16':
            weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG16_WEIGHTS,
                cache_subdir='models', file_hash='6d6bbae143d832006294945121d1f1fc')
        elif self.backbone_name == 'vgg16':
            weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG19_WEIGHTS,
                cache_subdir='models', file_hash='253f8cb515780f3b799900260a226db6')
        return weights_path

    def validate(self):
        allowed_backbone_names = ['vgg16', 'vgg19']

        if self.backbone_name not in allowed_backbone_names:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name, allowed_backbone_names))

    def preprocess_image(self, inputs):
        if self.backbone_name == 'vgg16':
            return vgg16.preprocess_input(inputs)
        elif self.backbone_name == 'vgg19':
            return vgg19.preprocess_input(inputs)