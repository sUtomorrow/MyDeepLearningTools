# -*- coding: utf-8 -*-
# @Time     : 9/25/19 9:07 AM
# @Author   : lty
# @File     : callbacks

import numpy as np
import torch
import cv2

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_overlap(boxes, query_boxes):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps



def Evaluate(model, generator, device, num_classes, iou_threshold=0.5, score_threshold=0.05, max_detections=100, random_valid_samples=500, verbose=False):
    with torch.no_grad():
        test_loss = 0
        rpn_reg_loss = 0
        rpn_cls_loss = 0
        reg_loss = 0
        cls_loss = 0
        if random_valid_samples:
            random_valid_idx_list = np.random.choice(len(generator), size=random_valid_samples, replace=False)
        else:
            random_valid_idx_list = [i for i in range(len(generator))]
        all_detections = [[None for _ in range(num_classes)] for _ in range(len(random_valid_idx_list))]
        all_annotations = [[None for _ in range(num_classes)] for _ in range(len(random_valid_idx_list))]
        for i, valid_idx in enumerate(random_valid_idx_list):
            data, batch_gt_boxes, batch_labels = generator[valid_idx]

            data = data.unsqueeze(0).to(device)
            batch_gt_boxes = torch.from_numpy(batch_gt_boxes).unsqueeze(0).to(device)
            batch_labels = torch.from_numpy(batch_labels).unsqueeze(0).to(device).long()
            bboxes, scores, labels, rpn_regression_loss, rpn_classification_loss, regression_loss, classification_loss = model(data, batch_gt_boxes, batch_labels)

            bboxes = bboxes.cpu().numpy() # [x1, y1, x2, y2]
            scores = scores.cpu().numpy() # scores
            labels = labels.cpu().numpy()

            # only deal with foreground
            mask = scores[0, :] > score_threshold

            # select those scores
            bboxes = bboxes[0][mask]
            scores = scores[0][mask]
            labels = labels[0][mask]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            image_bboxes = bboxes[scores_sort, :]
            image_scores = scores[scores_sort]
            image_labels = labels[scores_sort]

            # print(image_bboxes[:10])
            # print(image_scores[:10])
            # print(image_labels[:10])

            image_detections = np.concatenate([image_bboxes, np.expand_dims(image_scores, axis=1)], axis=1)

            gt_boxes  = batch_gt_boxes.cpu().numpy()[0, :, :4]
            gt_labels = batch_labels.cpu().numpy()[0, :]

            for label_idx in range(1, num_classes + 1):
                all_annotations[i][label_idx - 1] = gt_boxes[gt_labels == label_idx, :] #gt_boxes.copy()
                all_detections[i][label_idx - 1]  = image_detections[image_labels == label_idx, :]

                # print('label',label_idx, 'all_annotations', all_annotations[i][label_idx - 1].astype(np.int32))
                # print('label',label_idx, 'all_detections', all_detections[i][label_idx - 1].astype(np.int32))

            if verbose:
                image = data.cpu().numpy()[0, :, :, :]
                image = ((np.transpose(image, (1, 2, 0)) + 1) * 127).astype(np.uint8)
                for bbox, score in zip(bboxes, scores):
                    if score > 0.1:
                        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
                cv2.imshow('image', image)
                k = cv2.waitKey()
                if k == ord('q'):
                    exit()
                elif k == ord('c'):
                    verbose = False

            test_loss += rpn_regression_loss + rpn_classification_loss + regression_loss + classification_loss
            rpn_reg_loss += rpn_regression_loss
            rpn_cls_loss += rpn_classification_loss
            reg_loss += regression_loss
            cls_loss += classification_loss

        average_precisions = {}
        # process detections and annotations
        for label_idx in range(1, num_classes+1):
            # only deal with label 1
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(len(random_valid_idx_list)):
                detections = all_detections[i][label_idx - 1]
                annotations = all_annotations[i][label_idx - 1]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label_idx] = 0, 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = _compute_ap(recall, precision)
            average_precisions[label_idx] = average_precision, num_annotations


        test_loss    /= len(random_valid_idx_list)
        rpn_reg_loss /= len(random_valid_idx_list)
        rpn_cls_loss /= len(random_valid_idx_list)
        reg_loss     /= len(random_valid_idx_list)
        cls_loss     /= len(random_valid_idx_list)

        return average_precisions, test_loss.item(), rpn_reg_loss.item(), rpn_cls_loss.item(), reg_loss.item(), cls_loss.item()
