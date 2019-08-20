# -*- coding: utf-8 -*-
# @Time     : 5/25/19 10:42 AM
# @Author   : lty
# @File     : anchors

import tensorflow as tf
import keras
import numpy as np

default_anchor_params = {
    'sizes': [128],
    'ratios': [0.5, 1, 2.0],
    'scales': [0.5, 1., 2.0]
}

class PriorAnchor(keras.layers.Layer):
    def __init__(self, anchor_size_idx, feature_level, anchor_params=default_anchor_params, **kwargs):
        self.anchor_param  = anchor_params if anchor_params else default_anchor_params
        self.ratio         = np.array(self.anchor_param['ratios'], np.float32)
        self.scale         = np.array(self.anchor_param['scales'], np.float32)
        self.size          = self.scale * self.anchor_param['sizes'][anchor_size_idx]

        self.feature_level = feature_level

        self.anchor_size = np.tile(self.size[..., np.newaxis], (1, 2))
        self.anchor_size = np.reshape(np.tile(self.anchor_size, [1, len(self.ratio)]), (self.size.shape[0] * self.ratio.shape[0], 2))
        scale_factor     = np.tile(self.ratio, [len(self.size)])
        scale_factor     = np.stack([np.sqrt(scale_factor), 1 / np.sqrt(scale_factor)], axis=-1)
        self.anchor_size = self.anchor_size * scale_factor

        self.anchor_num = len(self.scale) * len(self.ratio)

        self.anchor_shift_tensor = tf.cast(np.concatenate([-self.anchor_size / 2, self.anchor_size - self.anchor_size / 2], axis = -1), tf.float32)
        self.anchor_shift_tensor = tf.reshape(self.anchor_shift_tensor, [1, 1, 1, self.anchor_num, 4])

        super(PriorAnchor, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        input_shape = tf.shape(inputs)
        batch_size  = input_shape[0]
        input_h     = input_shape[1]
        input_w     = input_shape[2]

        x, y = tf.meshgrid(tf.range(input_w), tf.range(input_h))

        x = tf.expand_dims(x, axis=-1)
        y = tf.expand_dims(y, axis=-1)

        point_grid = (tf.cast(keras.backend.concatenate([x, y, x, y]), tf.float32) + 0.5) * (2. ** self.feature_level)
        point_grid = tf.expand_dims(tf.expand_dims(point_grid, axis = -2), axis=0)
        point_grid = tf.tile(point_grid, [batch_size, 1, 1, self.anchor_num, 1])

        anchor_shift_tensor = tf.tile(self.anchor_shift_tensor, [batch_size, input_h, input_w, 1, 1])

        prior_anchor = tf.add(point_grid, anchor_shift_tensor)
        # prior_anchor = prior_anchor / tf.cast(2 ** self.feature_level, tf.float32)

        # anchor box coordinate should be [x1, y1, x2, y2], with shape: [batch_size, point_num, anchor_num, 4]
        return tf.reshape(prior_anchor, [batch_size, -1, 4])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], None, 4)


if __name__ == '__main__':
    """
    test the ProirAnchor layer
    """
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    feature_maps_shape = [None, None, None, 3]
    anchor_params = {'size': [4], 'ratio': [0.5, 1], 'scale': [1.0, 2.0]}
    feature_maps = np.zeros((10, 2, 2, 3), np.float32)

    with tf.Session() as session:
        feature_maps_tf = tf.placeholder(tf.float32, shape=feature_maps_shape)
        prior_anchor = PriorAnchor(0, 0, anchor_params=anchor_params)(feature_maps_tf)
        result = session.run(prior_anchor, feed_dict={feature_maps_tf: feature_maps})

    print(result[0])

    # output should be:
    # [[-0.91421354 -2.328427    1.9142135   3.328427  ]
    #  [-1.5        -1.5         2.5         2.5       ]
    #  [-2.328427   -5.156854    3.328427    6.156854  ]
    #  [-3.5        -3.5         4.5         4.5       ]
    #  [ 0.08578646 -2.328427    2.9142137   3.328427  ]
    #  [-0.5        -1.5         3.5         2.5       ]
    #  [-1.3284271  -5.156854    4.3284273   6.156854  ]
    #  [-2.5        -3.5         5.5         4.5       ]
    #  [-0.91421354 -1.3284271   1.9142135   4.3284273 ]
    #  [-1.5        -0.5         2.5         3.5       ]
    #  [-2.328427   -4.156854    3.328427    7.156854  ]
    #  [-3.5        -2.5         4.5         5.5       ]
    #  [ 0.08578646 -1.3284271   2.9142137   4.3284273 ]
    #  [-0.5        -0.5         3.5         3.5       ]
    #  [-1.3284271  -4.156854    4.3284273   7.156854  ]
    #  [-2.5        -2.5         5.5         5.5       ]]