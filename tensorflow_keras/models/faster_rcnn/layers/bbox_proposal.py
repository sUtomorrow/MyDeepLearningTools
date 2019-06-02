# -*- coding: utf-8 -*-
# @Time     : 5/30/19 1:01 PM
# @Author   : lty
# @File     : proposal

import tensorflow as tf
import tensorflow.keras as keras

class BboxProposal(keras.layers.Layer):

    def __init__(self, bbox_num=300,**kwargs):
        self.bbox_num = bbox_num
        super(BboxProposal, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        #TODO
        pass

    def compute_output_shape(self, input_shape):
        return [input_shape[0][0], self.bbox_num, 4]