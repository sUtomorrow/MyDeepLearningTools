# -*- coding: utf-8 -*-
# @Time     : 5/24/19 8:56 AM
# @Author   : lty
# @File     : DetectionDataGenerator

import torch
from torch.utils.data import Dataset

class DetectionDataGenerator(Dataset):
    def __init__(self, data_process_func_list, data_annotations2input_outputs, seed=10086, **kwargs):
        """
        :param data_process_func_list: list, list of the data augmentation functions, take (data, annotation) as input and return (data, annotation) after augmentation
        :param data_annotations2inputs_outputs: function, change the format of data and annotations to the input and target outputs for training model
        :param seed: random seed
        :param kwargs: args to initial the torch.utils.data.Dataset class
        """
        self.data_process_func_list = data_process_func_list
        self.seed = seed

        self.data_annotations2input_outputs = data_annotations2input_outputs

        # used to index label idx and label name
        self._class_name2idx = {}
        self._class_idx2name = {}

        super(DetectionDataGenerator, self).__init__(**kwargs)

    def __len__(self):
        """return number of data group"""
        raise NotImplementedError("__len__ function not implemented")

    def __getitem__(self, item_idx):
        """return data and label in a batch"""
        return self.compute_input_output(item_idx)

    def has_class_name(self, class_name):
        if class_name in self._class_name2idx:
            return True
        else:
            return False

    def has_class_idx(self, class_idx):
        if class_idx in self._class_idx2name:
            return True
        else:
            return False

    def class_name2idx(self, class_name):
        return self._class_name2idx[class_name]

    def class_idx2name(self, class_idx):
        return self._class_idx2name[class_idx]

    def load_data(self, data_idx):
        """load a data by data_idx"""
        raise NotImplementedError("load_data function not implemented")

    def load_annotations(self, data_idx):
        """load annotations by data_idx, return a list of annotation: [[x1, y1, x2, y2], ...]"""
        raise NotImplementedError("load_annotations function not implemented")

    def compute_input_output(self, data_idx):
        """
        1.load datas and annotations in one batch
        2.data augmentation for every (data, annotations) pair
        3.transform datas and annotations to input data and output label format
        """
        data = self.load_data(data_idx)
        annotations = self.load_annotations(data_idx)

        if self.data_process_func_list is not None:
            for data_process_func in self.data_process_func_list:
                # print(data.shape)
                data, annotations = data_process_func(data, annotations)

        return self.data_annotations2input_outputs(data, annotations)