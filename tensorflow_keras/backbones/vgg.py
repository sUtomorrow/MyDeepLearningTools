# -*- coding: utf-8 -*-
# @Time     : 5/25/19 10:40 AM
# @Author   : lty
# @File     : vgg

import keras
from keras.utils.data_utils import get_file
from keras.applications import vgg16, vgg19
from .backbone import Backbone

VGG16_WEIGHTS_PATH   = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
VGG16_WEIGHTS_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

VGG19_WEIGHTS_PATH   = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
VGG19_WEIGHTS_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

class VggBackbone(Backbone):
    def __init__(self, backbone_name, inputs=None, inputs_shape=None, include_top=False, classes=None, **kwargs):
        self.backbone_name = backbone_name
        self.include_top   = include_top
        self.classes       = classes

        if inputs is None:
            self.inputs = keras.layers.Input(shape=inputs_shape)
        else:
            self.inputs = inputs

        super(VggBackbone, self).__init__(**kwargs)

    def build(self):
        if self.backbone_name == 'vgg16':
            self._model = vgg16.VGG16(self.include_top, None, input_tensor=self.inputs, classes=self.classes)
            self._outputs = [self._model.get_layer('block%d_pool' % block_idx).output for block_idx in range(1, 6)]
            self._feature_levels = [level for level in range(1, 6)]
        elif self.backbone_name == 'vgg19':
            self._model = vgg19.VGG19(self.include_top, None, input_tensor=self.inputs, classes=self.classes)
            self._outputs = [self._model.get_layer('block%d_pool' % block_idx).output for block_idx in range(1, 6)]
            self._feature_levels = [level for level in range(1, 6)]
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(self.backbone_name))

        if self.include_top:
            self.outputs.append(self._model.get_layer('predictions').output)
            self.feature_levels.append(-1)

    def download_weights(self, cache_dir=None):
        weights_path = None
        if self.backbone_name == 'vgg16':
            if self.include_top:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',VGG16_WEIGHTS_PATH,
                    cache_subdir='models', file_hash='64373286793e3c8b2b4e3219cbf3544b', cache_dir=cache_dir)
            else:
                weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG16_WEIGHTS_NO_TOP,
                    cache_subdir='models', file_hash='6d6bbae143d832006294945121d1f1fc', cache_dir=cache_dir)
        elif self.backbone_name == 'vgg19':
            if self.include_top:
                weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', VGG19_WEIGHTS_PATH,
                    cache_subdir='models', file_hash='cbe5617147190e668d6c5d5026f83318', cache_dir=cache_dir)
            else:
                weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', VGG19_WEIGHTS_NO_TOP,
                    cache_subdir='models', file_hash='253f8cb515780f3b799900260a226db6', cache_dir=cache_dir)
        return weights_path

    def validate(self):
        allowed_backbone_names = ['vgg16', 'vgg19']

        if self.backbone_name not in allowed_backbone_names:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone_name, allowed_backbone_names))

    def load_weights(self, weight_path, by_name=True):
        self._model.load_weights(weight_path, by_name)

    def preprocess_image(self, inputs):
        if self.backbone_name == 'vgg16':
            return vgg16.preprocess_input(inputs)
        elif self.backbone_name == 'vgg19':
            return vgg19.preprocess_input(inputs)

if __name__ == '__main__':
    # test backbone
    vgg_backbone = VggBackbone('vgg19', None, (256, 256, 3), include_top=True, classes=10)
    outputs = vgg_backbone.outputs
    feature_levels = vgg_backbone.feature_levels
    for feature_level, output in zip(feature_levels, outputs):
        print(output.name, output.shape, 'level:', feature_level)