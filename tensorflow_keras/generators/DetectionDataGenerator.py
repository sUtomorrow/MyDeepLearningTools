# -*- coding: utf-8 -*-
# @Time     : 5/24/19 8:56 AM
# @Author   : lty
# @File     : DetectionDataGenerator

import numpy as np
import keras

class DetectionDataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, shuffle, data_aug_func_list, group_datas2inputs, group_annotations2outputs, seed=10086, **kwargs):
        """
        :param batch_size: int, size of batch data
        :param shuffle: bool, shuffle data on epoch end
        :param data_aug_func_list: list, list of the data augmentation functions, take (data, annotation) as input and return (data, annotation) after augmentation
        :param group_datas2inputs: function, change the format of a group of datas to the input data for training model
        :param group_annotations2outputs: function, change the format of a group of annotations to the output label for training model
        :param seed: random seed
        :param kwargs: args to initial the keras.utils.Sequence class
        """
        self.batch_size          = batch_size
        self.shuffle             = shuffle
        self.data_aug_func_list  = data_aug_func_list
        self.seed                = seed
        self.group_datas2inputs       = group_datas2inputs
        self.group_annotations2outputs = group_annotations2outputs

        # used to index label idx and label name
        self._label_name2idx = {}
        self._label_idx2name = {}

        np.random.seed(seed)

        self.groups = self.group_data()

        super(DetectionDataGenerator, self).__init__(**kwargs)

    def __len__(self):
        """return number of data group"""
        return len(self.groups)

    def __getitem__(self, item_idx):
        """return data and label in a batch"""
        return self.compute_input_output(item_idx)

    def has_label_name(self, label_name):
        if label_name in self._label_name2idx:
            return True
        else:
            return False

    def has_label_idx(self, label_idx):
        if label_idx in self._label_idx2name:
            return True
        else:
            return False

    def label_name2idx(self, label_name):
        return self._label_name2idx[label_name]

    def label_idx2name(self, label_idx):
        return self._label_idx2name[label_idx]

    def size(self):
        """return number of data"""
        raise NotImplementedError("size function not implemented")

    def group_data(self):
        """group data by data order index"""
        order = list(range(self.size()))
        if self.shuffle:
            np.random.shuffle(order)

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

    def load_data(self, data_idx):
        """load a data by data_idx"""
        raise NotImplementedError("load_data function not implemented")

    def load_group_datas(self, group_idx):
        """load datas in one batch"""
        return [self.load_data(data_idx) for data_idx in self.groups[group_idx]]

    def load_annotations(self, data_idx):
        """load annotations by data_idx, return a list of annotation: [[x1, y1, x2, y2], ...]"""
        raise NotImplementedError("load_annotations function not implemented")

    def load_group_annotations(self, group_idx):
        """load annotations in one batch"""
        return [self.load_annotations(data_idx) for data_idx in self.groups[group_idx]]

    def compute_input_output(self, group_idx):
        """
        1.load datas and annotations in one batch
        2.data augmentation for every (data, annotations) pair
        3.transform datas and annotations to input data and output label format
        """
        group_datas = self.load_group_datas(group_idx)
        group_annotations = self.load_group_annotations(group_idx)

        if self.data_aug_func_list is not None:
            for idx, data, annotations in enumerate(zip(group_datas, group_annotations)):
                for data_aug_func in self.data_aug_func_list:
                    data, annotations = data_aug_func(data, annotations)
                group_datas[idx] = data
                group_annotations[idx] = annotations

        inputs  = self.group_datas2inputs(group_datas, group_annotations)
        outputs = self.group_annotations2outputs(group_datas, group_annotations)

        return inputs, outputs

    def on_epoch_end(self):
        """callback function on every epoch end"""
        if self.shuffle:
            self.groups = self.group_data()