# -*- coding: utf-8 -*-
# @Time     : 6/11/19 6:50 PM
# @Author   : lty
# @File     : train

import sys
import argparse


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