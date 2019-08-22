# -*- coding: utf-8 -*-
# @Time     : 8/20/19 4:37 PM
# @Author   : lty
# @File     : utils

import numpy as np
import cv2

def group_datas_annotations2inputs_outputs(training=False, max_gts=200):
    def _group_datas_annotations2inputs_outputs(group_datas, group_annotations):
        image_inputs = np.array(group_datas)
        if training:
            gt_boxes_inputs = np.zeros((0, max_gts, 5))
            gt_class_idxes_inputs = np.zeros((0, max_gts, 2))
            for annotations in group_annotations:
                gt_boxes = []
                gt_class_idxes = []
                for bbox, class_idx in zip(annotations['bboxes'], annotations['class_idxes']):
                    gt_boxes.append(bbox + [1,])
                    gt_class_idxes.append([class_idx, 1])

                gt_boxes = np.array(gt_boxes, dtype=np.float32)
                gt_class_idxes = np.array(gt_class_idxes, dtype=np.float32)

                if len(gt_boxes) < max_gts:
                    gt_boxes = np.pad(gt_boxes, ((0, max_gts - len(gt_boxes)), (0, 0)), mode='constant', constant_values=0)
                    gt_class_idxes = np.pad(gt_class_idxes, ((0, max_gts - len(gt_class_idxes)), (0, 0)), mode='constant', constant_values=0)
                elif len(gt_boxes) > max_gts:
                    indeices = [True for _ in range(max_gts)] + [False for _ in range(len(gt_boxes) - max_gts)]
                    gt_boxes = gt_boxes[indeices, :]
                    gt_class_idxes = gt_class_idxes[indeices, :]
                # print(gt_boxes.shape)
                # print(gt_class_idxes.shape)
                gt_boxes_inputs = np.append(gt_boxes_inputs, gt_boxes[np.newaxis, ...], axis=0)
                gt_class_idxes_inputs = np.append(gt_class_idxes_inputs, gt_class_idxes[np.newaxis, ...], axis=0)

            return {'image_inputs': image_inputs,
                    'gt_boxes_inputs': gt_boxes_inputs,
                    'gt_class_idxes_inputs': gt_class_idxes_inputs}, None
        else:
            return {'image_inputs': image_inputs}, None
    return _group_datas_annotations2inputs_outputs

