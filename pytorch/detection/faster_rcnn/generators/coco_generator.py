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
        super(CocoGenerator, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.annotation_file_path = annotation_file_path
        self.coco = COCO(self.annotation_file_path)

        cats = self.coco.loadCats(self.coco.getCatIds()[10:12])
        self.class_names = tuple(['__background__'] + [c['name'] for c in cats])

        self._class_name2idx = dict(list(zip(self.class_names, list(range(len(self.class_names))))))
        self._class_idx2name = dict(list(zip(list(range(len(self.class_names))), self.class_names)))

        self.name2coco_cat_id = dict(list(zip([c['name'] for c in cats], self.coco.getCatIds()[10:12])))
        self.coco_cat_id2name = dict(list(zip(self.coco.getCatIds()[10:12], [c['name'] for c in cats])))

        print('initial classes:', self._class_idx2name)

        self.data_path_list, self.annotations_list = self._parse_data_from_coco()

    def __len__(self):
        return len(self.data_path_list)

    def _parse_data_from_coco(self):
        print('parsing data from coco')
        image_infos = [self.coco.loadImgs(img_id)[0] for img_id in sorted(self.coco.getImgIds())]
        image_path_list = [] # [os.path.join(self.data_dir, image_info['file_name']) for image_info in image_infos]
        annotations_list = []

        for image_info in image_infos:
            annotations = {
                'class_idxes': [],
                'bboxes': []
            }
            coco_annotationIds = self.coco.getAnnIds(imgIds=image_info['id'])
            coco_annotations = self.coco.loadAnns(coco_annotationIds)
            if len(coco_annotations) == 0:
                continue

            for coco_annotation in coco_annotations:
                if coco_annotation['bbox'][2] < 1 or coco_annotation['bbox'][3] < 1:
                    # skip some invalid annotation
                    continue
                # class_idx  = coco_annotation['category_id']
                class_name = self.coco.loadCats(ids=coco_annotation['category_id'])[0]['name']

                if class_name not in self._class_name2idx:
                    continue

                x1 = coco_annotation['bbox'][0]
                y1 = coco_annotation['bbox'][1]
                x2 = x1 + coco_annotation['bbox'][2]
                y2 = y1 + coco_annotation['bbox'][3]

                class_idx = self._class_name2idx[class_name]
                annotations['class_idxes'].append(class_idx)
                annotations['bboxes'].append([x1, y1, x2, y2])

            if len(annotations['bboxes']):
                annotations_list.append(annotations)
                image_path_list.append(os.path.join(self.data_dir, image_info['file_name']))

        print('data parsed')
        # print(class_idx2name)
        return image_path_list, annotations_list

    def load_data(self, data_idx):
        img = cv2.imread(self.data_path_list[data_idx])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_annotations(self, data_idx):
        return deepcopy(self.annotations_list[data_idx])