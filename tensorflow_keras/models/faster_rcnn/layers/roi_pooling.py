# -*- coding: utf-8 -*-
# @Time     : 5/24/19 1:13 PM
# @Author   : lty
# @File     : roi_pooling

import tensorflow as tf
import keras

class RoiPooling(keras.layers.Layer):
    def __init__(self, pooling_h, pooling_w, **kwargs):
        self.pooling_h = pooling_h
        self.pooling_w = pooling_w
        super(RoiPooling, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        '''
        return the output shape of ROI Pooling layer
        '''
        feature_map_shape, roi_shape = input_shape
        assert(feature_map_shape[0] == roi_shape[0])
        batch_size = feature_map_shape[0]
        n_rois = roi_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooling_h, self.pooling_w, n_channels)

    @staticmethod
    def _pool_roi(feature_map, roi, pooling_h, pooling_w):
        '''
        Applies single ROI on single image
        '''
        feature_map_h = tf.shape(feature_map)[0]
        feature_map_w = tf.shape(feature_map)[1]
        feature_map_c = tf.shape(feature_map)[2]


        w_start = tf.maximum(tf.cast(tf.round(roi[0]), tf.int32), tf.cast(0, tf.int32))
        h_start = tf.maximum(tf.cast(tf.round(roi[1]), tf.int32), tf.cast(0, tf.int32))
        w_end   = tf.minimum(tf.cast(tf.round(roi[2]), tf.int32), feature_map_w)
        h_end   = tf.minimum(tf.cast(tf.round(roi[3]), tf.int32), feature_map_h)

        # 我在github上看到一些代码直接把这个截取出来resize到输出大小了???这个也叫roi_pooling???
        roi_region = feature_map[h_start : h_end, w_start : w_end, :]

        roi_h = tf.cast(h_end - h_start, tf.float32)
        roi_w = tf.cast(w_end - w_start, tf.float32)

        h_step = tf.cast(roi_h, tf.float32) / tf.cast(pooling_h, tf.float32)
        w_step = tf.cast(roi_w, tf.float32) / tf.cast(pooling_w, tf.float32)
        
        # 注意这里的h_step和w_step可能小于1,这个时候pooling操作实际上将roi区域的特征值重复了
        areas = tf.cast(
            [
                [
                    i * h_step,
                    j * w_step,
                    (i + 1) * h_step if i + 1 < pooling_h else roi_h,
                    (j + 1) * w_step if j + 1 < pooling_w else roi_w
                ]
                for j in range(pooling_w) for i in range(pooling_h)
            ],
            tf.int32)

        def pooling_area(x):
            return tf.reduce_max(roi_region[x[0] : x[2], x[1] : x[3], :], axis = [0, 1])

        areas = tf.stop_gradient(areas)
        pooling_result = tf.map_fn(pooling_area, areas, dtype=tf.float32)

        pooling_result = tf.reshape(pooling_result, (pooling_h, pooling_w, feature_map_c))
        return pooling_result

    def call(self, inputs, **kwargs):
        '''
        roi should be [x1, y1, x2, y2], shape: [batch_size, roi_number, 4]
        '''
        feature_map = inputs[0]
        rois = inputs[1]
        batch_size = tf.shape(feature_map)[0]
        def batch_map_func(index):
            return tf.map_fn(lambda x: RoiPooling._pool_roi(feature_map[index], x, self.pooling_h, self.pooling_w), rois[index], dtype=tf.float32)

        return tf.map_fn(batch_map_func, tf.range(batch_size), dtype=tf.float32)


if __name__ == '__main__':
    """
    test the RoiPooling layer
    """
    import os
    import numpy as np

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Define parameters
    batch_size = 1000
    img_height = 200
    img_width = 100
    n_channels = 20
    n_rois = 2
    pooled_height = 3
    pooled_width = 7
    # Create feature map input
    feature_maps_shape = [None, img_height, img_width, n_channels]
    feature_maps_tf = tf.placeholder(tf.float32, shape=feature_maps_shape)
    feature_maps_shape[0] = batch_size
    feature_maps_np = np.ones(feature_maps_shape, dtype='float32')
    feature_maps_np[:, img_height - 1, img_width - 3, 0] = 50
    print(f"feature_maps_np.shape = {feature_maps_np.shape}")
    # Create batch size
    roiss_tf = tf.placeholder(tf.float32, shape=(None, n_rois, 4))
    roiss_np = np.asarray([[[50., 40., 70., 80.], [0.0, 0.0, 100.0, 200.0]]], dtype='float32')
    roiss_np = np.tile(roiss_np, (batch_size, 1, 1))
    print(f"roiss_np.shape = {roiss_np.shape}")
    # Create layer
    roi_layer = RoiPooling(pooled_height, pooled_width)
    pooled_features = roi_layer([feature_maps_tf, roiss_tf])
    print(f"output shape of layer call = {pooled_features.shape}")
    # Run tensorflow session
    with tf.Session() as session:
        result = session.run(pooled_features, feed_dict={feature_maps_tf: feature_maps_np, roiss_tf: roiss_np})

    print(f"result.shape = {result.shape}")
    print(f"first  roi embedding=\n{result[0,0,:,:,0]}")
    print(f"second roi embedding=\n{result[0,1,:,:,0]}")

    # result should be
    # feature_maps_np.shape = (1000, 200, 100, 20)
    # roiss_np.shape = (1000, 2, 4)
    # output shape of layer call = (?, 2, 3, 7, 20)
    # result.shape = (1000, 2, 3, 7, 20)
    # first  roi embedding=
    # [[1. 1. 1. 1. 1. 1. 1.]
    #  [1. 1. 1. 1. 1. 1. 1.]
    #  [1. 1. 1. 1. 1. 1. 1.]]
    # second roi embedding=
    # [[ 1.  1.  1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.  1. 50.]]