# -*- coding: utf-8 -*-
# @Time     : 6/2/19 4:13 PM
# @Author   : lty
# @File     : utils

import tensorflow as tf
import keras

class BoxClip(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(BoxClip, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        images = inputs[0]
        bboxes = inputs[1]
        image_shape = tf.shape(images)
        width = keras.backend.cast(image_shape[2], dtype=keras.backend.floatx())
        height = keras.backend.cast(image_shape[1], dtype=keras.backend.floatx())
        x1 = tf.clip_by_value(bboxes[:, :, 0], 0., width)
        y1 = tf.clip_by_value(bboxes[:, :, 1], 0., height)
        x2 = tf.clip_by_value(bboxes[:, :, 2], 0., width)
        y2 = tf.clip_by_value(bboxes[:, :, 3], 0., height)

        return keras.backend.stack([x1, y1, x2, y2], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[1]