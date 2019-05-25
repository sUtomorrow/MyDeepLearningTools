# -*- coding: utf-8 -*-
# @Time     : 5/23/19 9:34 PM
# @Author   : lty
# @File     : model

default_anchor_params = {
    'size': [128, 256, 512],
    'ratio': [0.5, 1, 2.0],
}

def FasterRCNN(backbone='vgg16', anchor_params=default_anchor_params):
    #TODO
    pass