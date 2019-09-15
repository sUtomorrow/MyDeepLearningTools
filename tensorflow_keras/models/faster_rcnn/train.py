# -*- coding: utf-8 -*-
# @Time     : 6/11/19 6:50 PM
# @Author   : lty
# @File     : train

import os
import tensorflow as tf
import keras
from model import FasterRcnn
from generators.coco_generator import CocoGenerator
from generators.data_process import random_transform_generator, data_aug_func, resize_image_func, image_process_func
from generators.utils import group_datas_annotations2inputs_outputs
from generators.debug import show_data
from utils import model_utils

train_config = {
    'anchor_params':{
        'sizes': [256],
        'ratios': [0.5, 1, 2.0],
        'scales': [0.5, 1.0, 2.0]
    },
    'model_params':{
        'BackboneName': 'vgg16',
        'ImageInputShape': (512, 512, 3),
        'MaxGTNum':200,
        'RpnPositiveIou': 0.5,
        'RpnNegativeIou': 0.3,
        'FrcnnPositiveIou': 0.8,
        'FrcnnNegativeIou': 0.5,
        'RoiPoolingW': 6,
        'RoiPoolingH': 6,
        'BboxProposalNum': 200, # image的大小和anchor params的设置，必须要产生多于BboxProposalNum的anchor个数
        'RegionProposalFilters': 512,
        'ClassNum': 2,
    },
    'gpu'              : '4,5',
    'imagenet_backbone': True, # load backbone weight with imagenet pretrain
    'backbone_weights'  : None, # load backbone weight from a file path, and ignore imagenet pretrain weight, if None: do not load
    'rpn_weights'       : None, # load rpn weigth from a file path, if None: do not load
    'rcnn_weights'     : None, # load faster-rcnn weight from a file path, if None: do not load
    'weight_save_dir'  : None, # the model weight save dir after train, if None: do not save model weight
    'train': True,
    'init_lr':1e-4,
    'lr_momentum':0.9,
    'loss_weights':{
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "rcnn_class_loss": 1.,
        "rcnn_bbox_loss": 1.
    },
    'workers': 8,
    'max_queue_size': 32,
    'initial_epoch': 0,
    'train_epoch': 100,
    'train_batch_size': 16,
    'clipnorm': 10.,
    'weight_decay': None,
    'rpn_weight_file_name': 'rpn_model',
    'frcnn_weight_file_name': 'faster_rcnn_model',
    'train_data_dir': '/mnt/data4/lty/data/coco/train2017',
    'valid_data_dir': '/mnt/data4/lty/data/coco/val2017',
    'train_annotations': '/mnt/data4/lty/data/coco/annotations/instances_train2017.json',
    'valid_annotations': '/mnt/data4/lty/data/coco/annotations/instances_val2017.json'
}

def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

if __name__ == '__main__':
    debug = False
    if debug:
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

        data_process_func_list = [data_aug_func(transform_generator, 'linear', 'constant', 0.),
                                        resize_image_func(train_config['model_params']['ImageInputShape'][:2], 'linear')
                                        ]


        valid_generator = CocoGenerator(
            data_dir=train_config['valid_data_dir'],
            annotation_file_path=train_config['valid_annotations'],
            batch_size=train_config['train_batch_size'],
            shuffle=True,
            data_process_func_list=data_process_func_list,
            group_datas_annotations2inputs_outputs=group_datas_annotations2inputs_outputs(training=True, max_gts=
            train_config['model_params']['MaxGTNum']),
        )
        show_data(valid_generator, with_annotatins=True)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = train_config['gpu']
        keras.backend.tensorflow_backend.set_session(get_session())

        image_preprocess_func, backbone_model, rpn_model, rcnn_model, faster_rcnn_inference, faster_rcnn_training = FasterRcnn(
            config=train_config,
            train=True,
            imagenet_backbone=train_config['imagenet_backbone'],
            backbone_weights=train_config['backbone_weights'],
            rpn_weights=train_config['rpn_weights'],
            rcnn_weights=train_config['rcnn_weights'],
        )

        loss_names = ["rpn_reg_loss", "rpn_cls_loss", "frcnn_reg_loss", "frcnn_cls_loss"]
        faster_rcnn_training = model_utils.compile(faster_rcnn_training,
                                                   train_config['init_lr'],
                                                   train_config['lr_momentum'],
                                                   train_config['clipnorm'],
                                                   train_config['weight_decay'],
                                                   loss_names,
                                                   train_config['loss_weights'])

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

        train_data_process_func_list = [data_aug_func(transform_generator, 'linear', 'constant', 0.),
                                    resize_image_func(train_config['model_params']['ImageInputShape'][:2], 'linear'),
                                    image_process_func(image_preprocess_func(train_config['model_params']['BackboneName']))]

        valid_data_process_func_list = [resize_image_func(train_config['model_params']['ImageInputShape'][:2], 'linear'),
                                    image_process_func(image_preprocess_func(train_config['model_params']['BackboneName']))]

        train_generator = CocoGenerator(
            data_dir=train_config['train_data_dir'],
            annotation_file_path=train_config['train_annotations'],
            batch_size=train_config['train_batch_size'],
            shuffle=True,
            data_process_func_list=train_data_process_func_list,
            group_datas_annotations2inputs_outputs=group_datas_annotations2inputs_outputs(training=True, max_gts=train_config['model_params']['MaxGTNum']),
        )

        valid_generator = CocoGenerator(
            data_dir=train_config['valid_data_dir'],
            annotation_file_path=train_config['valid_annotations'],
            batch_size=train_config['train_batch_size'],
            shuffle=True,
            data_process_func_list=valid_data_process_func_list,
            group_datas_annotations2inputs_outputs=group_datas_annotations2inputs_outputs(training=True, max_gts=
            train_config['model_params']['MaxGTNum']),
        )


        callback_list = []


        faster_rcnn_training.fit_generator(
            train_generator,
            epochs=train_config['train_epoch'],
            steps_per_epoch=len(train_generator) // train_config['train_batch_size'],
            verbose=1,
            initial_epoch=train_config['initial_epoch'],
            validation_data=valid_generator,
            validation_steps=len(valid_generator) // train_config['train_batch_size'],
            workers=train_config['workers'],
            use_multiprocessing=True,
            max_queue_size=train_config['max_queue_size'],
            callbacks=callback_list
        )



