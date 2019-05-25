# -*- coding: utf-8 -*-
# @Time     : 5/25/19 10:40 AM
# @Author   : lty
# @File     : vgg

import tensorflow as tf
import tensorflow.keras as keras
from .backbone import Backbone

class VggBackbone(Backbone):
    def __init__(self, backbone_name, inputs=None, inputs_shape=None):
        self.backbone_name = backbone_name
        if inputs is None:
            self.inputs = keras.Input(shape=inputs_shape)
        else:
            self.inputs = inputs

    def get_outputs(self):
        pass

    def download_weights(self):
        pass

    def validate(self):
        pass

    def preprocess_image(self, inputs):
        pass

    def get_custom_objects(self):
        pass