# -*- coding: utf-8 -*-
# @Time     : 9/20/19 9:06 AM
# @Author   : lty
# @File     : test

import torch

# indices = torch.randperm(3)
#
# print(indices)

indices = torch.zeros((2, 5), dtype=torch.long)

a = torch.rand((2, 3))

print(a)
print(-torch.log(torch.nn.functional.softmax(a, dim=-1)))
l = torch.nn.functional.cross_entropy(a, indices, reduction='none')
print(l)

# b = torch.rand((10, 4))
#
# print(b)

# print(a)
# a = a[indices[:3], :]
# print(a)
#
# a = a > 0.4
# b = b > 0.4
#
# print(torch.nonzero(a))
# print(torch.nonzero(b))
#
#
# b = b[a > 1]
# a = a[a > 1]
#
# print(b)
# print(b.size(1))