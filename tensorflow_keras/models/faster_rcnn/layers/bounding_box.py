# -*- coding: utf-8 -*-
# @Time     : 6/2/19 11:31 AM
# @Author   : lty
# @File     : bounding_box

import tensorflow as tf
import tensorflow.keras as keras

class BoundingBox(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BoundingBox, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ prior anchor should be [x1, y1, x2, y2],
            regression should be [tx, ty, tw, th]

            ax = (x1 + x2) / 2
            ay = (y1 + y2) / 2
            aw = x2 - x1
            ah = y2 - y1

            bx = ax + tx * aw
            by = ay + ty * ah
            bw = aw * exp(tw)
            bh = ah * exp(th)

            bx1 = bx - bw / 2
            by1 = by - bh / 2
            bx2 = bx1 + bw
            by2 = by1 + bh
            return [bx1, by1, bx2, by2]
        """
        prior_anchor = inputs[0]
        regression   = inputs[1]

        if len(prior_anchor.shape) == 2:
            # all input in one batch use same anchor
            prior_anchor = tf.expand_dims(prior_anchor, axis=0)

        anchor_x = (prior_anchor[:, :, 2] + prior_anchor[:, :, 0]) / 2
        anchor_y = (prior_anchor[:, :, 3] + prior_anchor[:, :, 1]) / 2

        anchor_w = prior_anchor[:, :, 2] - prior_anchor[:, :, 0]
        anchor_h = prior_anchor[:, :, 3] - prior_anchor[:, :, 1]

        bbox_x = anchor_x + regression[:, :, 0] * anchor_w
        bbox_y = anchor_y + regression[:, :, 1] * anchor_h
        bbox_w = anchor_w * tf.exp(regression[:, :, 2])
        bbox_h = anchor_h * tf.exp(regression[:, :, 3])

        bbox_x1 = bbox_x - bbox_w / 2
        bbox_y1 = bbox_y - bbox_h / 2
        bbox_x2 = bbox_x1 + bbox_w
        bbox_y2 = bbox_y1 + bbox_h

        return tf.stack([bbox_x1, bbox_y1, bbox_x2, bbox_y2], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

if __name__ == '__main__':
    """
    test the BoundingBox layer
    """
    from anchors import PriorAnchor
    import os
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    regressions_shape = [None, None, None, 4]
    anchor_params = {'size': [4], 'ratio': [1]}
    regression_np = np.zeros((10, 2, 2, 4), np.float32)
    regression_np[0, 0, 0, 0] = 1
    regression_np[0, 0, 0, 1] = 1
    regression_np[0, 0, 0, 2] = 0
    regression_np[0, 0, 0, 3] = 0
    feature_level = 1
    with tf.Session() as session:
        regression_tf = tf.placeholder(tf.float32, shape=regressions_shape)
        prior_anchor = PriorAnchor(feature_level, anchor_param=anchor_params)(regression_tf)
        regression_reshape = keras.layers.Reshape((-1, 4))(regression_tf)
        bbox = BoundingBox()([prior_anchor, regression_reshape])
        anchor, result = session.run([prior_anchor, bbox], feed_dict={regression_tf: regression_np})

    print(anchor[0])
    print(result[0])

    # result should be
    # [[-1. -1.  3.  3.]
    #  [ 1. -1.  5.  3.]
    #  [-1.  1.  3.  5.]
    #  [ 1.  1.  5.  5.]]
    # [[ 3.  3.  7.  7.]
    #  [ 1. -1.  5.  3.]
    #  [-1.  1.  3.  5.]
    #  [ 1.  1.  5.  5.]]



