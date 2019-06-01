# -*- coding: utf-8 -*-
# @Time     : 5/30/19 1:01 PM
# @Author   : lty
# @File     : proposal

import tensorflow as tf
import tensorflow.keras as keras

class Proposal(keras.layers.Layer):

    def __init__(self, **kwargs):

        super(Proposal, self).__init__(**kwargs)