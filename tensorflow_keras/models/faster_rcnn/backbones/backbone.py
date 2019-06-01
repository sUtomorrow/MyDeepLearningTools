# -*- coding: utf-8 -*-
# @Time     : 5/25/19 4:09 PM
# @Author   : lty
# @File     : backbone


class Backbone(object):
    def __init__(self):
        # if using custom objects, the custom_objects should be updated
        self.custom_objects = {}

        self.validate()

    def build_network(self):
        """ built backbone network and set self.outputs and self.feature_level
        """
        raise NotImplementedError('build_network method not implemented')

    def get_outputs(self):
        if hasattr(self, 'outputs'):
            return self.outputs
        else:
            self.build_network()
            return self.outputs

    def get_feature_level(self):
        """return the downsample times of model's outputs
        """
        if hasattr(self, 'feature_level'):
            return self.feature_level
        else:
            self.build_network()
            return self.feature_level

    def get_custom_objects(self):
        """ return custom objects dict for this backbone
        """
        return self.custom_objects

    def download_weights(self):
        """ Download model weights and returns path to weights file.
        """
        raise NotImplementedError('download_weights method not implemented.')

    def load_weights(self, weight_path):
        """ load model weights
        """
        raise NotImplementedError('load_weights method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')