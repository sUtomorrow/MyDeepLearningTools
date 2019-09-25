# -*- coding: utf-8 -*-
# @Time     : 9/25/19 9:07 AM
# @Author   : lty
# @File     : callbacks

import numpy as np
import torch

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



def EvaluateRPN(model, data_loader, device, iou_threshold=0.3, score_threshold=0.05, max_detections=100, random_valid_samples=500):
    generator = data_loader.dataset
    with torch.no_grad():
        test_loss = 0
        reg_loss = 0
        cls_loss = 0

        random_valid_idx_list = np.random.choice(len(generator), size=random_valid_samples, replace=False)

        all_detections = [[None for _ in range(1)] for _ in range(len(random_valid_idx_list))]
        all_annotations = [[None for _ in range(1)] for _ in range(len(random_valid_idx_list))]

        for i, valid_idx in enumerate(random_valid_idx_list):
            data, batch_gt_boxes, batch_labels = generator[valid_idx]

            data = data.unsqueeze(0).to(device)
            batch_gt_boxes = torch.tensor(batch_gt_boxes, device=device).unsqueeze(0)
            batch_labels = torch.tensor(batch_labels, device=device).unsqueeze(0)
            rpn_bboxes, rpn_classifications, rpn_regression_loss, rpn_classification_loss = model(data, batch_gt_boxes, batch_labels)
            rpn_scores, rpn_labels = torch.max(rpn_classifications, dim=2)

            bboxes = rpn_bboxes.cpu().numpy() # [x1, y1, x2, y2]
            scores = rpn_scores.cpu().numpy() # scores
            labels = rpn_labels.cpu().numpy() # 1 for foreground, 0 for background

            # only deal with foreground
            indices = np.where((scores[0, :] > score_threshold) & (labels[0, :] == 1))[0]

            # select those scores
            scores = scores[0][indices]

            # find the order with which to sort the scores
            scores_sort = np.argsort(-scores)[:max_detections]

            # select detections
            image_bboxes = bboxes[0, indices[scores_sort], :]
            image_scores = scores[scores_sort]
            image_labels = labels[0, indices[scores_sort]]

            image_detections = np.concatenate([image_bboxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

            gt_boxes = batch_gt_boxes.cpu().numpy()[0, :, :4]
            gt_labels = batch_labels.cpu().numpy()[0, :]
            gt_boxes = gt_boxes[gt_labels != -1, :] # for rpn, all boxes are foreground
            all_annotations[i][0] = gt_boxes.copy()

            for label_idx in range(1):
                # only deal with label 1
                all_detections[i][label_idx] = image_detections[image_detections[:, -1] == label_idx + 1, :-1]

            test_loss += rpn_regression_loss + rpn_classification_loss
            reg_loss += rpn_regression_loss
            cls_loss += rpn_classification_loss

        average_precisions = {}
        # process detections and annotations
        for label_idx in range(1):
            # only deal with label 1
            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(len(random_valid_idx_list)):
                detections = all_detections[i][label_idx]
                annotations = all_annotations[i][label_idx]
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

        rpn_average_precision = average_precisions[0][0]

        test_loss /= len(random_valid_idx_list)
        reg_loss /= len(random_valid_idx_list)
        cls_loss /= len(random_valid_idx_list)

        return rpn_average_precision, test_loss.item(), reg_loss.item(), cls_loss.item()
