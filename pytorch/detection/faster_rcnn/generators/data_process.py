# -*- coding: utf-8 -*-
# @Time     : 8/21/19 10:14 AM
# @Author   : lty
# @File     : random_transform
import os
import sys
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import generators  # noqa: F401
    __package__ = "generators"

import numpy as np
import cv2
import threading


def _random_vector(min, max, prng):
    """ Construct a random vector between min and max.
    Args
        min: the minimum value for each component
        max: the maximum value for each component
    """
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return prng.uniform(min, max)


def _rotation(angle):
    """ Construct a homogeneous 2D rotation matrix.
        Args
            angle: the angle in radians
        Returns
            the rotation matrix as 3 by 3 numpy array
        """
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


def _random_rotation(rotation_ratio, min_rotation, max_rotation, prng):
    """ Construct a random rotation between min_rotation and min_rotation.
    Args
        rotation_ratio: probability (0 to 1) of rotation
        min_rotation:  a scalar for the minimum angle in radians
        max_rotation:  a scalar for the maximum angle in radians
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 rotation matrix
    """
    if prng.choice([True, False], p=[rotation_ratio, 1-rotation_ratio]):
        return _rotation(prng.uniform(min_rotation, max_rotation))
    else:
        return _rotation(0.)


def _translate(translation):
    """ Construct a homogeneous 2D translation matrix.
    # Arguments
        translation: the translation 2D vector
    # Returns
        the translation matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])


def _random_translate(translation_ratio, min_translation, max_translation, prng):
    """ Construct a random 2D translation between min_translation and max_translation.
    Args
        translation_ratio: probability (0 to 1) of translation
        min_translation:  a 2D vector with the minimum translation for each dimension
        max_translation:  a 2D vector with the maximum translation for each dimension
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 translation matrix
    """
    if prng.choice([True, False], p=[translation_ratio, 1-translation_ratio]):
        return _translate(_random_vector(min_translation, max_translation, prng))
    else:
        return _translate([0., 0.])


def _shear(angle):
    """ Construct a homogeneous 2D shear matrix.
    Args
        angle: the shear angle in radians
    Returns
        the shear matrix as 3 by 3 numpy array
    """
    return np.array([
        [1, -np.sin(angle), 0],
        [0,  np.cos(angle), 0],
        [0, 0, 1]
    ])


def _random_shear(shear_ratio, min_shear, max_shear, prng):
    """ Construct a random 2D shear matrix with shear angle between -max and max.
    Args
        shear_ratio: probability (0 to 1) of shear
        min_shear:  the minimum shear angle in radians.
        max_shear:  the maximum shear angle in radians.
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 shear matrix
    """
    if prng.choice([True, False], p=[shear_ratio, 1 - shear_ratio]):
        return _shear(prng.uniform(min_shear, max_shear))
    else:
        return _shear(0.)


def _scaling(factor):
    """ Construct a homogeneous 2D scaling matrix.
    Args
        factor: a 2D vector for X and Y scaling
    Returns
        the zoom matrix as 3 by 3 numpy array
    """
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def _random_scaling(scaling_ratio, min_scaling, max_scaling, prng):
    """ Construct a random 2D scale matrix between -max and max.
    Args
        scaling_ratio: probability (0 to 1) of scaling
        min_scaling:  a 2D vector containing the minimum scaling factor for X and Y.
        max_scaling:  a 2D vector containing The maximum scaling factor for X and Y.
        prng: the pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 scaling matrix
    """
    if prng.choice([True, False], p=[scaling_ratio, 1 - scaling_ratio]):
        return _scaling(_random_vector(min_scaling, max_scaling, prng))
    else:
        return _scaling([1., 1.])


def _random_flip(flip_x_ratio, flip_y_ratio, prng):
    """ Construct a transformation randomly containing X/Y flips (or not).
    Args
        flip_x_ratio: The chance that the result will contain a flip along the X axis.
        flip_y_ratio: The chance that the result will contain a flip along the Y axis.
        prng:          The pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 transformation matrix
    """
    flip_x = prng.choice([1, 0], p=[flip_x_ratio, 1 - flip_x_ratio])
    flip_y = prng.choice([1, 0], p=[flip_y_ratio, 1 - flip_y_ratio])
    return _scaling((1 - 2 * flip_x, 1 - 2 * flip_y))


def random_transform(
    rotation_ratio=0.5,
    min_rotation=0,
    max_rotation=0,
    translation_ratio=0.5,
    min_translation=(0, 0),
    max_translation=(0, 0),
    shear_ratio=0.5,
    min_shear=0,
    max_shear=0,
    scaling_ratio=0.5,
    min_scaling=(1., 1.),
    max_scaling=(1., 1.),
    flip_x_ratio=0,
    flip_y_ratio=0,
    prng=None
):
    """ Create a random transform generator.

    Uses a dedicated, newly created, properly seeded PRNG by default instead of the global DEFAULT_PRNG.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
    as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    Args
        rotation_ratio:     The probability (0 to 1) of rotation
        min_rotation:       The minimum rotation in radians for the transform as scalar.
        max_rotation:       The maximum rotation in radians for the transform as scalar.
        translation_ratio:  The probability (0 to 1) of translation
        min_translation:    The minimum translation for the transform as 2D column vector.
        max_translation:    The maximum translation for the transform as 2D column vector.
        shear_ratio:        The probability (0 to 1) of shear
        min_shear:          The minimum shear angle for the transform in radians.
        max_shear:          The maximum shear angle for the transform in radians.
        shear_scaling:      The probability (0 to 1) of scaling
        min_scaling:        The minimum scaling for the transform as 2D column vector.
        max_scaling:        The maximum scaling for the transform as 2D column vector.
        flip_x_ratio:       The probability (0 to 1) that a transform will contain a flip along X direction.
        flip_y_ratio:       The probability (0 to 1) that a transform will contain a flip along Y direction.
        prng:               The pseudo-random number generator to use.
    """
    return np.linalg.multi_dot([
        _random_rotation(rotation_ratio, min_rotation, max_rotation, prng),
        _random_translate(translation_ratio, min_translation, max_translation, prng),
        _random_shear(shear_ratio, min_shear, max_shear, prng),
        _random_scaling(scaling_ratio, min_scaling, max_scaling, prng),
        _random_flip(flip_x_ratio, flip_y_ratio, prng)
    ])


def change_transform_origin(transform, center):
    """ Create a new transform representing the same transformation,
        only with the origin of the linear part changed.
    Args
        transform: the transformation matrix
        center: the new origin of the transformation
    Returns
        _translation(center) * transform * _translation(-center)
    """
    center = np.array(center)
    return np.linalg.multi_dot([_translate(center), transform, _translate(-center)])


def _adjust_transform_for_image(transform, image):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    # Move the origin of transformation.
    return change_transform_origin(transform, (0.5 * width, 0.5 * height))


def cvBorderMode(fill_mode):
    if fill_mode == 'constant':
        return cv2.BORDER_CONSTANT
    if fill_mode == 'nearest':
        return cv2.BORDER_REPLICATE
    if fill_mode == 'reflect':
        return cv2.BORDER_REFLECT_101
    if fill_mode == 'wrap':
        return cv2.BORDER_WRAP


def cvInterpolation(interpolation):
    if interpolation == 'nearest':
        return cv2.INTER_NEAREST
    if interpolation == 'linear':
        return cv2.INTER_LINEAR
    if interpolation == 'cubic':
        return cv2.INTER_CUBIC
    if interpolation == 'area':
        return cv2.INTER_AREA
    if interpolation == 'lanczos4':
        return cv2.INTER_LANCZOS4


def transform_bbox(transform, bbox):
    """ Apply a transformation to an axis aligned bounding box.

    The result is a new boxes in the same coordinate system as the original boxes.
    The new bbox contains all corner points of the original bbox after applying the given transformation.

    Args
        transform: The transformation to apply.
        x1:        The minimum x value of the AABB.
        y1:        The minimum y value of the AABB.
        x2:        The maximum x value of the AABB.
        y2:        The maximum y value of the AABB.
    Returns
        The new bbox as tuple (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    # Transform all 4 corners of the AABB.
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1,  1,  1,  1 ],
    ])

    # Extract the min and max corners again.
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]


class thread_safe_generator(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.lock = threading.Lock()

    def __next__(self):
        with self.lock:
            return random_transform(**self.params)

    def __iter__(self):
        return next(self)


def random_transform_generator(prng=None, **kwargs):
    """ Create a random transform generator.

    Uses a dedicated, newly created, properly seeded PRNG by default instead of the global DEFAULT_PRNG.

    The transformation consists of the following operations in this order (from left to right):
      * rotation
      * translation
      * shear
      * scaling
      * flip x (if applied)
      * flip y (if applied)

    Note that by default, the data generators in `keras_retinanet.preprocessing.generators` interpret the translation
    as factor of the image size. So an X translation of 0.1 would translate the image by 10% of it's width.
    Set `relative_translation` to `False` in the `TransformParameters` of a data generator to have it interpret
    the translation directly as pixel distances instead.

    Args
        min_rotation:    The minimum rotation in radians for the transform as scalar.
        max_rotation:    The maximum rotation in radians for the transform as scalar.
        min_translation: The minimum translation for the transform as 2D column vector.
        max_translation: The maximum translation for the transform as 2D column vector.
        min_shear:       The minimum shear angle for the transform in radians.
        max_shear:       The maximum shear angle for the transform in radians.
        min_scaling:     The minimum scaling for the transform as 2D column vector.
        max_scaling:     The maximum scaling for the transform as 2D column vector.
        flip_x_chance:   The chance (0 to 1) that a transform will contain a flip along X direction.
        flip_y_chance:   The chance (0 to 1) that a transform will contain a flip along Y direction.
        prng:            The pseudo-random number generator to use.
    """

    if prng is None:
        # RandomState automatically seeds using the best available method.
        prng = np.random.RandomState()

    return thread_safe_generator(prng=prng, **kwargs)


def data_aug_func(random_transform_generator, interpolation='linear', fill_mode='constant', c_value=0.):
    def _data_aug_func(data, annotations):
        transform = _adjust_transform_for_image(next(random_transform_generator), data)
        data = cv2.warpAffine(
            data,
            transform[:2, :],
            dsize=(data.shape[1], data.shape[0]),
            flags=cvInterpolation(interpolation),
            borderMode=cvBorderMode(fill_mode),
            borderValue=c_value,
        )
        for index in range(len(annotations['bboxes'])):
            annotations['bboxes'][index] = transform_bbox(transform, annotations['bboxes'][index])
        return data, annotations
    return _data_aug_func


def resize_image_func(d_size=(512, 512), interpolation='linear'):
    def _resize_image_func(data, annotations):
        fy = d_size[0] / data.shape[0]
        fx = d_size[1] / data.shape[1]
        data = cv2.resize(data, dsize=None, fx=fx, fy=fy, interpolation=cvInterpolation(interpolation))
        for index in range(len(annotations['bboxes'])):
            annotations['bboxes'][index][0] *= fx
            annotations['bboxes'][index][1] *= fy
            annotations['bboxes'][index][2] *= fx
            annotations['bboxes'][index][3] *= fy
        return data, annotations
    return _resize_image_func


def image_process_func(image_process):
    # return a process function only process image
    def _image_process_func(data, annotations):
        data = image_process(data)
        return data, annotations
    return _image_process_func


if __name__ == '__main__':
    
    from .coco_generator import CocoGenerator
    from .utils import data_annotations2input_outputs
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
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
                                    resize_image_func((448, 448), 'linear'),
                                    image_process_func(transforms.ToTensor()),
                                    # image_process_func(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])),
                                    ]

    train_generator = CocoGenerator(
        data_dir='/mnt/data4/lty/data/coco/val2017',
        annotation_file_path='/mnt/data4/lty/data/coco/annotations/instances_val2017.json',
        data_process_func_list=train_data_process_func_list,
        data_annotations2input_outputs=data_annotations2input_outputs(max_gts=30),
    )

    data_loader = DataLoader(train_generator, batch_size=4, shuffle=True, num_workers=4)

    for inputs, gt_boxes, gt_class_idxes in data_loader:

        image = inputs[0, :, :, :].numpy()
        image = np.transpose(image, [1, 2, 0])
        gt_class_idxes = gt_class_idxes[0].numpy()
        gt_boxes       = gt_boxes[0].numpy()

        image = (image * 255).astype(np.uint8)
        # print(image.max())
        # print(image.min())
        for gt_box, gt_class_idx in zip(gt_boxes, gt_class_idxes):
            if gt_class_idx == -1:
                break
            print(gt_class_idx)
            print(gt_box)
            image = cv2.rectangle(image, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), color=(0, 0, 255), thickness=4)

        cv2.imshow('image', image)

        k = cv2.waitKey()
        if k == ord('q'):
            exit()
        # print(inputs.shape)
        # print(inputs.max())
        # print(inputs.min())
        # print(gt_boxes.shape)
        # print(gt_boxes)
        # print(gt_class_idxes.shape)
        # print(gt_class_idxes)
        # exit()

    
