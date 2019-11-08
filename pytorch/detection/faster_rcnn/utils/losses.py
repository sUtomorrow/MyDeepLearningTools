# -*- coding: utf-8 -*-
# @Time     : 9/18/19 10:48 AM
# @Author   : lty
# @File     : losses

import torch
import torch.nn.functional as F

def smooth_l1(regression, regression_target, sigma=2.0):
    """
    :param regression: the regression value of model with shape (BatchSize, H, W, A*4), [tx, ty, tw, th]
    :param regression_target: the regression target with shape (BatchSize, H*W*A, 5), [tx, ty, tw, th, weight]
        weight: 0 for ignore
    :return: smooth l1 loss
    """
    weight = regression_target[:, :, -1]
    regression_target = regression_target[:, :, :-1]

    # print('loss regression_target:', regression_target[weight > 0, :][:1, :])
    # print('loss regression:', regression[weight > 0, :][:1, :])

    diff = regression - regression_target

    abs_diff = torch.abs(diff)

    # print('abs_diff:', abs_diff[weight > 0, :][:4, :])

    smooth_sign = (abs_diff < 1./sigma).detach().float()
    #
    loss = (torch.pow(abs_diff, 2) * 0.5 * sigma * smooth_sign + (abs_diff - 0.5 / sigma) * (1. - smooth_sign))

    # print('loss:', loss[weight > 0, :][:4, :])

    loss = loss.mean(2) * weight
    # loss = abs_diff.mean(2) * weight

    norm = torch.tensor(max(weight.nonzero().size(0), 1), dtype=loss.dtype)
    # print('l1 norm', norm)
    loss = loss.sum() / norm
    # print('regression loss', loss)
    return loss


# def _smooth_l1_loss(bbox_pred, bbox_targets, sigma=1.0, dim=[1]):
#     sigma_2 = sigma ** 2
#
#     box_diff = bbox_pred - bbox_targets
#
#     in_box_diff = bbox_inside_weights * box_diff
#
#     abs_in_box_diff = torch.abs(in_box_diff)
#
#     smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
#
#     in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
#                   + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
#
#     out_loss_box = bbox_outside_weights * in_loss_box
#
#     loss_box = out_loss_box
#     for i in sorted(dim, reverse=True):
#         loss_box = loss_box.sum(i)
#     loss_box = loss_box.mean()
#     return loss_box

def cross_entropy_loss(classification, classification_target, num_classes=2):
    """
    :param classification       : the classification value of model with shape (BatchSize, H, W, A, num_classes)
    :param classification_target: the classification_target target with shape (BatchSize, H*W*A, 2), [class_idx, weight]
        weight: 0 for ignore
    :return: cross entropy loss
    """
    # batch_size     = classification.size(0)
    classification = classification#.contiguous().view(-1, num_classes)
    # print('classification.size()', classification.size())
    weights               = classification_target[:, :, -1]
    classification_target = classification_target[:, :, 0]#.contiguous()#.view(-1)

    # print('tags.size()', tags.size())
    # indices = torch.nonzero(weights).view(-1)
    # print('indices.size()', indices.size())
    # classification = classification[indices, :]
    # classification_target = classification_target[indices]
    # print('classification_target for train', classification_target[indices])

    # if classification.size(0) == 0:
    #     return torch.zeros(1)
    # print(classification.size(0))
    # print('classification.size()', classification.size())
    # print('classification_target.size()', classification_target.size())
    # print('classification_target[weight > 0][:1]', classification_target[weight > 0][:1])
    # print('classification[weight > 0, :][:1]', classification[weight > 0, :][:1])
    # loss = F.nll_loss(classification, classification_target)#.reshape_as(weight)

    batch_idx = classification_target.size(0)
    N = classification_target.size(1)
    classification = classification.view(batch_idx * N, num_classes)
    classification_target = classification_target.view(batch_idx * N, num_classes)
    loss = F.cross_entropy(classification, classification_target, reduction='none') #
    # print('loss.size()', loss.size())
    loss = loss.mean(2).reshape_as(weights)
    loss *= weights.type_as(loss)
    norm = torch.tensor(max(weights.nonzero().size(0), 1), dtype=loss.dtype)

    # print('ce norm', norm)
    loss = loss.sum() / norm
    # print('classification loss', loss)
    return loss


def focal_loss(classification, classification_target, num_classes, gamma = 2, class_weights=None):
    classification = classification.contiguous().view(-1, num_classes)
    weights = classification_target[:, :, -1].contiguous().view(-1)
    classification_target = classification_target[:, :, 0].contiguous().view(-1)

    logpt = F.log_softmax(classification, dim=1)
    pt = torch.exp(logpt)
    logpt = (1 - pt) ** gamma * logpt
    loss = F.nll_loss(logpt, classification_target, class_weights, reduction='none').reshape_as(weights)
    loss *= weights.type_as(loss)
    norm = torch.tensor(max(weights.nonzero().size(0), 1), dtype=loss.dtype)
    loss = loss.sum() / norm
    return loss

def cross_entropy_loss_show(classification, classification_target, num_classes=2):
    """
    :param classification       : the classification value of model with shape (BatchSize, H, W, A, num_classes)
    :param classification_target: the classification_target target with shape (BatchSize, H*W*A, 2), [class_idx, weight]
        weight: 0 for ignore
    :return: cross entropy loss
    """
    # batch_size     = classification.size(0)
    classification = classification.contiguous().view(-1, num_classes)
    # print('classification.size()', classification.size())
    weights               = classification_target[:, :, -1].contiguous().view(-1)
    classification_target = classification_target[:, :, 0].contiguous().view(-1)

    # print('tags.size()', tags.size())
    indices = torch.nonzero(weights).view(-1)
    # print('indices.size()', indices.size())
    # classification = classification[indices, :]
    print('classification_target for train', classification_target[indices])

    # if classification.size(0) == 0:
    #     return torch.zeros()
    # print(classification.size(0))
    # print('classification.size()', classification.size())
    # print('classification_target.size()', classification_target.size())
    # print('classification_target[weight > 0][:1]', classification_target[weight > 0][:1])
    # print('classification[weight > 0, :][:1]', classification[weight > 0, :][:1])
    # loss = F.nll_loss(classification, classification_target)#.reshape_as(weight)
    loss = F.cross_entropy(classification, classification_target, reduction='none').reshape_as(weights) #
    # print('loss.size()', loss.size())
    loss *= weights.type_as(loss)
    norm = torch.tensor(max(weights.nonzero().size(0), 1), dtype=loss.dtype)

    # print('ce norm', norm)
    loss = loss.sum() / norm
    # print('classification loss', loss)
    return loss


