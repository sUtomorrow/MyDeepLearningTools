# -*- coding: utf-8 -*-
# @Time     : 9/18/19 10:48 AM
# @Author   : lty
# @File     : losses

import torch

def smooth_l1(regression, regression_target, sigma=9.0):
    """
    :param regression: the regression value of model with shape (BatchSize, H, W, A*4), [tx, ty, tw, th]
    :param regression_target: the regression target with shape (BatchSize, H*W*A, 5), [tx, ty, tw, th, weight]
        weight: 0 for ignore
    :return: smooth l1 loss
    """
    weight = regression_target[:, :, -1]
    regression_target = regression_target[:, :, :-1]

    # print('regression_target:', regression_target[weight > 0, :][:10, :])
    # print('regression:', regression[weight > 0, :][:10, :])

    diff = regression - regression_target
    abs_diff = torch.abs(diff)
    smooth_sign = (abs_diff < 1./sigma).float()
    loss = (torch.pow(diff, 2) * 0.5 * sigma * smooth_sign + (abs_diff - 0.5 / sigma) * (1. - smooth_sign))
    loss = loss.sum(2) * weight

    norm = torch.tensor(max(weight.nonzero().size(0), 1), dtype=loss.dtype)

    loss = loss.sum() / norm
    return loss

def cross_entropy_loss(classification, classification_target):
    """
    :param classification       : the classification value of model with shape (BatchSize, H, W, A), [foreground,]
    :param classification_target: the classification_target target with shape (BatchSize, H*W*A, 2), [foreground, weight]
        weight: 0 for ignore
    :return: cross entropy loss
    """
    # batch_size     = classification.size(0)
    classification = classification.contiguous().view(-1, 2)

    weight                = classification_target[:, :, 1].contiguous().view(-1)
    classification_target = classification_target[:, :, 0].contiguous().view(-1)

    # print('classification_target:', classification_target[weight > 0][:10])
    # print('classification:', classification[weight > 0, :][:10, :])

    loss = torch.nn.functional.cross_entropy(classification, classification_target, reduction='none').reshape_as(weight)
    loss *= weight.type_as(loss)
    norm = torch.tensor(max(weight.nonzero().size(0), 1), dtype=loss.dtype)
    loss = loss.sum() / norm
    return loss




