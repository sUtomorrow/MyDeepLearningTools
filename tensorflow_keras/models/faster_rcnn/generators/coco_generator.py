# -*- coding: utf-8 -*-
# @Time     : 8/13/19 4:11 PM
# @Author   : lty
# @File     : coco_generator

import os
import cv2
from pycocotools.coco import COCO
from ....generators import DetectionDataGenerator

class CocoGenerator(DetectionDataGenerator):
    def __init__(self, data_dir, annotation_file_path, **kwargs):
        self.data_dir = data_dir
        self.annotation_file_path = annotation_file_path
        self.coco = COCO(self.annotation_file_path)
        self.data_path_list, self.annotations, self._label_idx2name, self._label_name2idx = self._parse_data_from_coco()
        super(CocoGenerator, self).__init__(**kwargs)

    def _parse_data_from_coco(self):
        print('parsing data from coco')
        image_infos = [self.coco.loadImgs(img_id) for img_id in sorted(self.coco.getImgIds())]
        image_path_list = [] # [os.path.join(self.data_dir, image_info['file_name']) for image_info in image_infos]
        labels_list = []
        bboxes_list = []
        label_idx2name = {}
        label_name2idx = {}
        for image_info in image_infos:
            coco_annotationIds = self.coco.getAnnIds(imgIds=image_info['id'])
            coco_annotations = self.coco.loadAnns(coco_annotationIds)
            image_path_list.append(os.path.join(self.data_dir, image_info['file_name']))
            labels = []
            bboxes = []
            for coco_annotation in coco_annotations:
                label_idx = coco_annotation['category_id'] - 1
                label_name = self.coco.loadCats(ids=label_idx+1)[0]
                if label_idx not in label_idx2name:
                    label_idx2name[label_idx] = label_name

                if label_name not in label_name2idx:
                    label_idx2name[label_name] = label_idx

                labels.append(label_idx)
                bboxes.append(coco_annotation['bbox'])
            labels_list.append(labels)
            bboxes_list.append(bboxes)
        annotations     = {
            'labels': labels_list,
            'bboxes': bboxes_list
        }
        print('data parsed')
        return image_path_list, annotations, label_idx2name, label_name2idx

    def size(self):
        return len(self.data_path_list)

    def load_data(self, data_idx):
        img = cv2.imread(self.data_path_list[data_idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_annotations(self, data_idx):
        return self.annotations[data_idx]



