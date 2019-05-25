# -*- coding: utf-8 -*-
# @Time     : 5/25/19 4:09 PM
# @Author   : lty
# @File     : backbone


class Backbone(object):

    def get_outputs(self):
        """ Faster R-CNN need noly one output from backbone model
        """
        raise NotImplementedError('get_outputs method not implemented')

    def get_custom_objects(self):
        """ return custom objects dict for this backbone
        """
        return {}

    def download_weights(self):
        """ Download model weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')