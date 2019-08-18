# -*- coding: utf-8 -*-
# @Time     : 6/11/19 6:50 PM
# @Author   : lty
# @File     : train

import sys
import argparse
import warnings
import numpy as np

from keras.engine.training_utils import iter_sequence_infinite
from keras import backend as K
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras import callbacks as cbks


train_config = {
    'gpu'              : '0',
    'backbone'         : 'vgg16',
    'imagenet_backbone': True, # load backbone weight with imagenet pretrain
    'backbone_weight'  : None, # load backbone weight from a file path, and ignore imagenet pretrain weight, if None: do not load
    'rpn_weight'       : None, # load rpn weigth from a file path, if None: do not load
    'frcnn_weight'     : None, # load faster-rcnn weight from a file path, if None: do not load
    'weight_save_dir'  : None, # the model weight save dir after train, if None: do not save model weight

    # step 1 train region proposal network
    'train_rpn': True,
    'rpn_weight_file_name': 'rpn_model',
    'rpn_train_init_lr': 1e-4,
    'rpn_train_epoch': 20,

    # step 2 train faster-rcnn model without region proposal network
    'train_frcnn': True,
    'frcnn_weight_file_name': 'faster_rcnn_model',
    'frcnn_train_init_lr': 1e-4,
    'frcnn_train_epoch': 20,

    # step 3 train total faster-rcnn model
    'train': True,


}


def fit_generator(model,
                  generator,
                  steps_per_epoch=None,
                  epochs=1,
                  verbose=1,
                  callbacks=None,
                  validation_data=None,
                  validation_steps=None,
                  class_weight=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0):
    """See docstring for `Model.fit_generator`."""
    wait_time = 0.01  # in seconds
    epoch = initial_epoch

    do_validation = bool(validation_data)
    model._make_train_function()
    if do_validation:
        model._make_test_function()

    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps_per_epoch is None:
        if is_sequence:
            steps_per_epoch = len(generator)
        else:
            raise ValueError('`steps_per_epoch=None` is only valid for a'
                             ' generator based on the '
                             '`keras.utils.Sequence`'
                             ' class. Please specify `steps_per_epoch` '
                             'or use the `keras.utils.Sequence` class.')

    # python 2 has 'next', 3 has '__next__'
    # avoid any explicit version checks
    val_gen = (hasattr(validation_data, 'next') or
               hasattr(validation_data, '__next__') or
               isinstance(validation_data, Sequence))
    if (val_gen and not isinstance(validation_data, Sequence) and
            not validation_steps):
        raise ValueError('`validation_steps=None` is only valid for a'
                         ' generator based on the `keras.utils.Sequence`'
                         ' class. Please specify `validation_steps` or use'
                         ' the `keras.utils.Sequence` class.')

    # Prepare display labels.
    out_labels = model.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

    # prepare callbacks
    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.stateful_metric_names)]
    if verbose:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=model.stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    enqueuer = None
    val_enqueuer = None

    try:
        if do_validation:
            if val_gen and workers > 0:
                # Create an Enqueuer that can be reused
                val_data = validation_data
                if isinstance(val_data, Sequence):
                    val_enqueuer = OrderedEnqueuer(
                        val_data,
                        use_multiprocessing=use_multiprocessing)
                    validation_steps = validation_steps or len(val_data)
                else:
                    val_enqueuer = GeneratorEnqueuer(
                        val_data,
                        use_multiprocessing=use_multiprocessing)
                val_enqueuer.start(workers=workers,
                                   max_queue_size=max_queue_size)
                val_enqueuer_gen = val_enqueuer.get()
            elif val_gen:
                val_data = validation_data
                if isinstance(val_data, Sequence):
                    val_enqueuer_gen = iter_sequence_infinite(val_data)
                    validation_steps = validation_steps or len(val_data)
                else:
                    val_enqueuer_gen = val_data
            else:
                # Prepare data for validation
                if len(validation_data) == 2:
                    val_x, val_y = validation_data
                    val_sample_weight = None
                elif len(validation_data) == 3:
                    val_x, val_y, val_sample_weight = validation_data
                else:
                    raise ValueError('`validation_data` should be a tuple '
                                     '`(val_x, val_y, val_sample_weight)` '
                                     'or `(val_x, val_y)`. Found: ' +
                                     str(validation_data))
                val_x, val_y, val_sample_weights = model._standardize_user_data(
                    val_x, val_y, val_sample_weight)
                val_data = val_x + val_y + val_sample_weights
                if model.uses_learning_phase and not isinstance(K.learning_phase(),
                                                                int):
                    val_data += [0.]
                for cbk in callbacks:
                    cbk.validation_data = val_data

        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    shuffle=shuffle)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if is_sequence:
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator

        callback_model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            for m in model.stateful_metric_functions:
                m.reset_states()
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            while steps_done < steps_per_epoch:
                generator_output = next(output_generator)

                if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))

                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
                # build batch logs
                batch_logs = {}
                if x is None or len(x) == 0:
                    # Handle data tensors support when no input given
                    # step-size = 1 for data tensors
                    batch_size = 1
                elif isinstance(x, list):
                    batch_size = x[0].shape[0]
                elif isinstance(x, dict):
                    batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)

                outs = model.train_on_batch(x, y,
                                            sample_weight=sample_weight,
                                            class_weight=class_weight)

                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                batch_index += 1
                steps_done += 1

                # Epoch finished.
                if steps_done >= steps_per_epoch and do_validation:
                    if val_gen:
                        val_outs = model.evaluate_generator(
                            val_enqueuer_gen,
                            validation_steps,
                            workers=0)
                    else:
                        # No need for try/except because
                        # data has already been validated.
                        val_outs = model.evaluate(
                            val_x, val_y,
                            batch_size=batch_size,
                            sample_weight=val_sample_weights,
                            verbose=0)
                    val_outs = to_list(val_outs)
                    # Same labels assumed.
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

                if callback_model.stop_training:
                    break

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callback_model.stop_training:
                break

    finally:
        try:
            if enqueuer is not None:
                enqueuer.stop()
        finally:
            if val_enqueuer is not None:
                val_enqueuer.stop()

    callbacks.on_train_end()
    return model.history

def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    parser.add_argument('--backbone',         help='Backbone model used by faster-rcnn.', default='vgg16', type=str)
    parser.add_argument('--batch-size',       help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi), split by ","', default="")
    parser.add_argument('--epochs',           help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',            help='Number of steps per epoch.', type=int, default=1024)
    parser.add_argument('--lr',               help='Learning rate.', type=float, default=1e-3)
    parser.add_argument('--train-step1',      help='train rpn.', action='store_true', default=False)
    parser.add_argument('--train-step2',      help='save boxes predicted by rpn.', action='store_true', default=False)
    parser.add_argument('--train-step3',      help='train faster-rcnn use saved boxes', action='store_true', default=False)
    parser.add_argument('--snapshot-path',    help='Path to store models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir',  help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',     help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',    help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--loss-alpha',
                        help='param alpha for focal loss', type = float, default = 0.25)
    parser.add_argument('--loss-gamma', help='param gamma for focal loss', type = float, default = 2.0)

    # Fit generator arguments
    parser.add_argument('--workers', help='Number of multiprocessing workers. To disable multiprocessing, set workers to 0', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit generator.', type=int, default=10)
    parser.add_argument('--test-froc', help='test froc cal', action='store_true')
    parser.add_argument('--online-aug', help='use online augementation', dest='online_aug', action='store_true', default = False)
    parser.add_argument('--random-mask-ratio', help='ratio of random mask augementation', type = float, default = 0)
    parser.add_argument('--elastic-trans-ratio', help='ratio of elastic transform', type = float, default = 0)
    parser.add_argument('--cutout-ratio', help='ratio of cutout', type=float, default=0)
    parser.add_argument('--contrast-decline-ratio', help='ratio of constrast decline ratio', type=float, default=0)
    parser.add_argument('--color-disturb-ratio', help='ratio of color disturb', type=float, default=0)
    return parser.parse_args(args)

def main(args):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)


if __name__ == '__main__':
    main()