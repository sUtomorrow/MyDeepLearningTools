#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/11 21:21
# @Author  : Lty
# @File    : rescale.py

import keras

class Rescale(keras.layers.Layer):
    def __init__(self, scale=1., **kwargs):
        self.scale = scale
        super(Rescale, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return inputs * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape