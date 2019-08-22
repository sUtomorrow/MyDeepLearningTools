# -*- coding: utf-8 -*-
# @Time     : 6/11/19 6:50 PM
# @Author   : lty
# @File     : train

from generators.coco_generator import CocoGenerator
from generators.data_process import random_transform_generator, data_aug_func, resize_image_func
from generators.utils import group_datas_annotations2inputs_outputs
from generators.debug import show_data

train_config = {
    'anchor_params':{
        'sizes': [256],
        'strides': [32],
        'ratios': [0.5, 1, 2.0],
        'scales': [0.5, 1.0, 2.0]
    },
    'model_params':{
        'BackboneName': 'vgg16',
        'InputShape': (None, None, 3),
        'RpnPositiveIou': 0.5,
        'RpnNegativeIou': 0.3,
        'FrcnnPositiveIou': 0.8,
        'FrcnnNegativeIou': 0.5,
        'RoiPoolingW': 6,
        'RoiPoolingH': 6,
        'BboxProposalNum': 300, # image的大小和anchor params的设置，必须要产生多于BboxProposalNum的anchor个数
        'RegionProposalFilters': 512,
        'ClassNum': 2,
    },
    'gpu'              : '0',
    'imagenet_backbone': True, # load backbone weight with imagenet pretrain
    'backbone_weight'  : None, # load backbone weight from a file path, and ignore imagenet pretrain weight, if None: do not load
    'rpn_weight'       : None, # load rpn weigth from a file path, if None: do not load
    'frcnn_weight'     : None, # load faster-rcnn weight from a file path, if None: do not load
    'weight_save_dir'  : None, # the model weight save dir after train, if None: do not save model weight
    'train': True,
    'rpn_weight_file_name': 'rpn_model',
    'frcnn_weight_file_name': 'faster_rcnn_model',
    'train_data_dir': '/mnt/data4/lty/data/coco/train2017',
    'valid_data_dir': '/mnt/data4/lty/data/coco/val2017',
    'train_annotations': '/mnt/data4/lty/data/coco/annotations/instances_train2017.json',
    'valid_annotations': '/mnt/data4/lty/data/coco/annotations/instances_val2017.json'
}



if __name__ == '__main__':
    transform_generator = random_transform_generator(
        rotation_ratio=0.5,
        min_rotation=-0.2,
        max_rotation=0.2,
        translation_ratio=0.5,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        shear_ratio=0.5,
        min_shear=-0.2,
        max_shear=0.2,
        scaling_ratio=0.5,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_ratio=0.5,
        flip_y_ratio=0.,
        prng=None
    )

    data_aug_func_list = [data_aug_func(transform_generator, 'linear', 'constant', 0.), resize_image_func((512, 512), 'linear')]

    train_generator = CocoGenerator(
        data_dir=train_config['valid_data_dir'],
        annotation_file_path=train_config['valid_annotations'],
        batch_size=1,
        shuffle=True,
        data_aug_func_list=data_aug_func_list,
        group_datas_annotations2inputs_outputs=group_datas_annotations2inputs_outputs(training=True, max_gts=200),
    )

    show_data(train_generator, True)
