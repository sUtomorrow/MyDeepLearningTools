# -*- coding: utf-8 -*-
# @Time     : 8/12/19 10:37 AM
# @Author   : lty
# @File     : ClassificationDataGenerator

import numpy as np
import keras

class ClassificationDataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size, shuffle, data_aug_func_list, group_datas2inputs, group_labels2outputs, seed=10086, **kwargs):
        """
        :param batch_size: int, size of batch data
        :param shuffle: bool, shuffle data on epoch end
        :param data_aug_func_list: list, list of the data augmentation functions, which take (data, label) as input and return (data, label) after augmentation
        :param group_datas2inputs: function, change the format of a group of datas to the input data for training model
        :param group_labels2outputs: function, change the format of a group of labels to the output label for training model
        :param seed: random seed
        :param kwargs: args to initial the keras.utils.Sequence class
        """
        self.batch_size          = batch_size
        self.shuffle             = shuffle
        self.data_aug_func_list  = data_aug_func_list
        self.seed                = seed
        self.group_datas2inputs  = group_datas2inputs
        self.group_labels2outputs = group_labels2outputs

        np.random.seed(seed)

        self.groups = self.group_data()

        super(ClassificationDataGenerator, self).__init__(**kwargs)

    def __len__(self):
        """return number of data group"""
        return len(self.groups)

    def __getitem__(self, item_idx):
        """return data and label in a batch"""
        return self.compute_input_output(item_idx)

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

    def load_labels(self, data_idx):
        """load labels by data_idx, return a list of label: [[x1, y1, x2, y2], ...]"""
        raise NotImplementedError("load_labels function not implemented")

    def load_group_labels(self, group_idx):
        """load labels in one batch"""
        return [self.load_labels(data_idx) for data_idx in self.groups[group_idx]]

    def compute_input_output(self, group_idx):
        """
        1.load datas and labels in one batch
        2.data augmentation for every (data, labels) pair
        3.transform datas and labels to input data and output label format
        """
        group_datas = self.load_group_datas(group_idx)
        group_labels = self.load_group_labels(group_idx)

        if self.data_aug_func_list is not None:
            for idx, data, labels in enumerate(zip(group_datas, group_labels)):
                for data_aug_func in self.data_aug_func_list:
                    data, labels = data_aug_func(data, labels)
                group_datas[idx] = data
                group_labels[idx] = labels

        inputs  = self.group_datas2inputs(group_datas, group_labels)
        outputs = self.group_labels2outputs(group_datas, group_labels)

        return inputs, outputs

    def on_epoch_end(self):
        """callback function on every epoch end"""
        if self.shuffle:
            self.groups = self.group_data()