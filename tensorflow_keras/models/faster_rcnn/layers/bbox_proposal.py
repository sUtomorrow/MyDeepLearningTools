# -*- coding: utf-8 -*-
# @Time     : 5/30/19 1:01 PM
# @Author   : lty
# @File     : proposal

import tensorflow as tf
import keras

class BboxProposal(keras.layers.Layer):
    def __init__(self, bbox_num=300, nms_threshold=0.5, **kwargs):
        self.bbox_num = bbox_num
        self.nms_threshold = nms_threshold
        super(BboxProposal, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        classifications, bboxes = inputs

        def _nms(params):
            scores, bboxes = params
            nms_indices = tf.image.non_max_suppression(bboxes, scores, max_output_size=self.bbox_num, iou_threshold=self.nms_threshold)
            bboxes = tf.gather(bboxes, nms_indices, axis = 0)

            pad_size = keras.backend.maximum(0, self.bbox_num - keras.backend.shape(bboxes)[0])
            bboxes = tf.pad(bboxes, [[0, pad_size], [0, 0]], constant_values=-1)

            return bboxes

        scores = classifications[:, :, 1] # the last dimension of classifications is in one-hot: [background, foreground]

        bboxes = tf.map_fn(
            _nms,
            elems=[scores, bboxes],
            dtype=tf.float32
        )
        return bboxes

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.bbox_num, 4)


if __name__ == '__main__':
    """
    test the BboxProposal layer
    """
    import os
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    bbox_np = np.array([[[0, 0, 20, 20], [0, 0, 20, 10], [0, 0, 50, 60], [0, 0, 40, 30]], [[0, 0, 20, 10], [0, 0, 30, 20], [0, 0, 50, 50], [0, 0, 100, 20]]], dtype=np.float32)

    classification_np = np.array([[[0.1, 0.9], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]], [[0.1, 0.9], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]]], dtype=np.float32)


    with tf.Session() as session:
        bbox_tf = tf.placeholder(tf.float32, shape=[None, None, 4])
        classification_tf = tf.placeholder(tf.float32, shape=[None, None, 2])

        bbox_proposal = BboxProposal(bbox_num=4, nms_threshold=0.3)([classification_tf, bbox_tf])
        result = session.run(bbox_proposal, feed_dict={bbox_tf: bbox_np, classification_tf:classification_np})

    print(result)

    # [[[  0.   0.  20.  20.]
    #   [  0.   0.  50.  60.]
    #   [ -1.  -1.  -1.  -1.]
    #   [ -1.  -1.  -1.  -1.]]
    #
    #  [[  0.   0.  20.  10.]
    #   [  0.   0. 100.  20.]
    #   [  0.   0.  50.  50.]
    #   [ -1.  -1.  -1.  -1.]]]