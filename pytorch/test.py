# -*- coding: utf-8 -*-
# @Time     : 9/20/19 9:06 AM
# @Author   : lty
# @File     : test

import torch
import numpy as np
from detection.faster_rcnn.utils.nms import _C

bbox_np = np.array([[0, 0, 20, 20], [0, 0, 20, 10], [0, 0, 50, 60], [0, 0, 40, 30]], dtype=np.float32)

score_np = np.array([0.9, 0.6, 0.3, 0.8], dtype=np.float32)

bboxes = torch.from_numpy(bbox_np)
scores = torch.from_numpy(score_np)


indices = _C.nms(bboxes, scores, 0.5)

print(bboxes[indices, :])
print(scores[indices])

# indices = torch.randperm(3)
#
# print(indices)

# indices = torch.zeros((2), dtype=torch.long)
# #
# a = torch.rand((4, 2, 4))
#
# print(torch.nonzero(a))

#
# print(a)
# print(-torch.log(torch.nn.functional.softmax(a, dim=-1)))
# l = torch.nn.functional.cross_entropy(a, indices, reduction='none')
# print(l)

# print([1, 2, 3, 4].index(10))

# b = torch.rand((4, 4))
#
# print(b)
# c = b[:, :2].contiguous().view(8)
# print(c)
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