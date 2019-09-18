# -*- coding: utf-8 -*-
# @Time     : 9/18/19 10:48 AM
# @Author   : lty
# @File     : losses

import torch

def smooth_l1(regression, regression_target, sigma=1.0):
    sigma_2 = sigma ** 2
    diff = regression - regression_target
    abs_diff = torch.abs(diff)
    #TODO: to complete the loss function 

