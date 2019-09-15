# -*- coding: utf-8 -*-
# @Time     : 9/15/19 9:49 AM
# @Author   : lty
# @File     : rpn

import torch

class RPN(torch.nn.Module):
    def __init__(self, in_channel, filters, anchor_num):
        super(RPN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, filters, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
        self.classification = torch.nn.Conv2d(filters, anchor_num, kernel_size=(1, 1), padding=(1, 1), stride=(1, 1))
        self.sigmoid = torch.nn.Sigmoid()
        self.regression = torch.nn.Conv2d(filters, anchor_num * 4, kernel_size=(1, 1), padding=(1, 1), stride=(1, 1))

    