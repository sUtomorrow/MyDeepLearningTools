# -*- coding: utf-8 -*-
# @Time     : 8/22/19 3:58 PM
# @Author   : lty
# @File     : debug

import numpy as np
import cv2

def draw_bboxes(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int32)
    for bbox in bboxes:
        if bbox[4] == 0:
            continue
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=2)
    return image


def show_data(data_generator, with_annotatins=True):
    for i in range(len(data_generator)):
        inputs, targets = data_generator[i]
        for batch_idx in range(len(inputs['image_inputs'])):
            image = cv2.cvtColor(inputs['image_inputs'][0], cv2.COLOR_RGB2BGR)
            if with_annotatins:
                image = draw_bboxes(image, inputs['gt_boxes_inputs'][batch_idx])
            cv2.imshow('image', image)
            # print(inputs['gt_boxes_inputs'][batch_idx, :10, :])  # .shape)#
            # print(inputs['gt_class_idxes_inputs'][batch_idx, :10, :])  # .shape)#
            k = cv2.waitKey()
            if k == ord('q'):
                exit()