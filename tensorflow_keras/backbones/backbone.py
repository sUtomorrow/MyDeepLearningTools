# -*- coding: utf-8 -*-
# @Time     : 5/25/19 4:09 PM
# @Author   : lty
# @File     : backbone


class Backbone(object):
    def __init__(self, **kwargs):
        #if using custom objects, the custom_objects should be updated in build
        self.custom_objects = {}

        self.validate()

    def build(self):
        """ build backbone network,update self.custom_objects and set properties as follow:
            self.outputs        : all output in a list
            self.feature_levels : down sample times of output in a list, if output is not feature map, use -1
        """
        raise NotImplementedError('build method not implemented')

    def get_outputs(self):
        """return the list of outputs
        """
        if hasattr(self, 'outputs'):
            return self.outputs
        else:
            self.build()
            return self.outputs

    def get_feature_level(self):
        """return the downsample times of model's outputs
        """
        if hasattr(self, 'feature_levels'):
            return self.feature_levels
        else:
            self.build()
            return self.feature_levels

    def get_custom_objects(self):
        """return custom objects dict for this backbone
        """
        return self.custom_objects

    def download_weights(self):
        """Download model weights and returns path to weights file.
        """
        raise NotImplementedError('download_weights method not implemented.')

    def load_weights(self, weight_path, by_name=True):
        """load model weights
        """
        raise NotImplementedError('load_weights method not implemented.')

    def validate(self):
        """Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')