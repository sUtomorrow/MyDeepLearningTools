# -*- coding: utf-8 -*-
# @Time     : 8/13/19 4:11 PM
# @Author   : lty
# @File     : coco_generator

import os
import cv2
from copy import deepcopy
from pycocotools.coco import COCO
from .DetectionDataGenerator import DetectionDataGenerator

class CocoGenerator(DetectionDataGenerator):
    def __init__(self, data_dir, annotation_file_path, **kwargs):
        self.data_dir = data_dir
        self.annotation_file_path = annotation_file_path
        self.coco = COCO(self.annotation_file_path)
        self.data_path_list, self.annotations_list, self._class_idx2name, self._class_name2idx = self._parse_data_from_coco()
        super(CocoGenerator, self).__init__(**kwargs)

    def _parse_data_from_coco(self):
        print('parsing data from coco')
        image_infos = [self.coco.loadImgs(img_id)[0] for img_id in sorted(self.coco.getImgIds())]
        image_path_list = [] # [os.path.join(self.data_dir, image_info['file_name']) for image_info in image_infos]
        annotations_list = []
        class_idx2name = {}
        class_name2idx = {}
        for image_info in image_infos:
            annotations = {
                'class_idxes': [],
                'bboxes': []
            }
            coco_annotationIds = self.coco.getAnnIds(imgIds=image_info['id'])
            coco_annotations = self.coco.loadAnns(coco_annotationIds)
            if len(coco_annotations) == 0:
                continue
            image_path_list.append(os.path.join(self.data_dir, image_info['file_name']))
            for coco_annotation in coco_annotations:
                if coco_annotation['bbox'][2] < 1 or coco_annotation['bbox'][3] < 1:
                    # skip some invalid annotation
                    continue

                x1 = coco_annotation['bbox'][0]
                y1 = coco_annotation['bbox'][1]
                x2 = x1 + coco_annotation['bbox'][2]
                y2 = y1 + coco_annotation['bbox'][3]

                class_idx = coco_annotation['category_id'] - 1
                class_name = self.coco.loadCats(ids=coco_annotation['category_id'])[0]['name']
                if class_idx not in class_idx2name:
                    class_idx2name[class_idx] = class_name

                if class_name not in class_name2idx:
                    class_name2idx[class_name] = class_idx

                annotations['class_idxes'].append(class_idx)
                annotations['bboxes'].append([x1, y1, x2, y2])
            annotations_list.append(annotations)

        print('data parsed')
        return image_path_list, annotations_list, class_idx2name, class_name2idx

    def size(self):
        return len(self.data_path_list)

    def load_data(self, data_idx):
        img = cv2.imread(self.data_path_list[data_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_annotations(self, data_idx):
        return deepcopy(self.annotations_list[data_idx])