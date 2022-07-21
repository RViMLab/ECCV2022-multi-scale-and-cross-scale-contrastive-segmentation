import numpy as np
from utils import to_numpy, CLASS_INFO


def get_confusion_matrix(prediction, target, existing_matrix=None):
    """Expects prediction logits (as output by network), and target as classes in single channel (as from data)"""
    prediction, target = to_numpy(prediction), to_numpy(target)
    num_classes = prediction.shape[1]  # prediction is shape NCHW, we want C (one-hot length of all classes)
    one_hots = np.eye(num_classes)
    prediction = np.moveaxis(prediction, 1, 0)  # Prediction is NCHW -> move C to the front to make it CNHW
    prediction = np.reshape(prediction, (num_classes, -1))  # Prediction is [C, N*H*W]
    prediction = np.argmax(prediction, 0)  # Prediction is now [N*H*W]
    one_hot_preds = one_hots[prediction]  # Prediction is now [N*H*W, C]
    one_hot_preds = np.moveaxis(one_hot_preds, 1, 0)  # Prediction is now [C, N*H*W]
    one_hot_targets = one_hots[target.reshape(-1)]  # Target is now [N*H*W, C]
    confusion_matrix = np.matmul(one_hot_preds, one_hot_targets).astype('i')  # [C, N*H*W] x [N*H*W, C] = [C, C]
    # Consistency check:
    assert(np.sum(confusion_matrix) == target.size)  # All elements summed equals all pixels in original target
    for i in range(num_classes):
        assert(np.sum(confusion_matrix[i]) == np.sum(prediction == i))  # Row of matrix equals class incidence in pred
        assert(np.sum(confusion_matrix[:, i]) == np.sum(target == i))  # Col of matrix equals class incidence in target
    if existing_matrix is not None:
        assert(existing_matrix.shape == confusion_matrix.shape)
        confusion_matrix += existing_matrix
    return confusion_matrix


def normalise_confusion_matrix(matrix, mode):
    if mode == 'row':
        row_sums = matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # to avoid division by 0. Safe, because if sum is 0, all elements have to be 0 too
        matrix = matrix / row_sums[:, np.newaxis]
    elif mode == 'col':
        col_sums = matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1  # to avoid division by 0. Safe, because if sum is 0, all elements have to be 0 too
        matrix = matrix / col_sums[np.newaxis, :]
    else:
        raise ValueError("Normalise confusion matrix: mode needs to be either 'row' or 'col'.")
    return matrix


def get_pixel_accuracy(confusion_matrix):
    """Pixel accuracies, adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch

    :param confusion_matrix: Confusion matrix with absolute values. Rows are predicted classes, columns ground truths
    :return: Overall pixel accuracy, pixel accuracy per class (PA / PAC in CaDISv2 paper)
    """
    pred_class_correct = np.diag(confusion_matrix)
    acc = np.sum(pred_class_correct) / np.sum(confusion_matrix)
    pred_class_sums = np.sum(confusion_matrix, axis=1)
    pred_class_sums[pred_class_sums == 0] = 1  # To avoid division by 0 problems. Safe because all elem = 0 when sum = 0
    acc_per_class = np.mean(pred_class_correct / pred_class_sums)
    return acc, acc_per_class


def get_mean_iou(confusion_matrix, experiment, categories=False, single_class=None):
    """Uses confusion matrix to compute mean iou. Confusion matrix computed by get_confusion_matrix: row indexes
    prediction class, column indexes ground truth class. Based on:
    github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py
    """
    assert experiment in [1, 2, 3], 'experiment must be in [1,2,3] instead got [{}]'.format(experiment)
    if single_class is not None:
        # compute miou for a single_class
        assert(not categories),\
            'when single_class is not None, category must be False instead got [{}]'.format(categories)
        assert(single_class in CLASS_INFO[experiment]),\
            'single_class must be {} instead got [{}]'.format(CLASS_INFO[experiment][1].keys(), single_class)
        return get_single_class_iou(confusion_matrix, experiment, single_class)
    elif categories:
        # compute miou for the classes of instruments and for the classes of anatomies
        # compute miou for all classes
        assert (single_class is None),\
            'when category is not None, single class must be None instead got [{}]'.format(single_class)
        miou_instruments = np.mean([get_single_class_iou(confusion_matrix, experiment, c)
                                    for c in CLASS_INFO[experiment][2]['instruments']])
        miou_anatomies = np.mean([get_single_class_iou(confusion_matrix, experiment, c)
                                  for c in CLASS_INFO[experiment][2]['anatomies']])
        miou = np.mean([get_single_class_iou(confusion_matrix, experiment, c)
                        for c in CLASS_INFO[experiment][1].keys()])
        return miou, miou_instruments, miou_anatomies
    else:
        # compute miou for all classes
        miou = np.mean([get_single_class_iou(confusion_matrix, experiment, c)
                        for c in CLASS_INFO[experiment][1].keys()])
        return miou


def get_single_class_iou(confusion_matrix, experiment, single_class):
    if single_class == 255:  # This is the 'ignore' class helpfully introduced in exp 2 and 3. Needs to NOT be 255 here
        single_class = confusion_matrix.shape[0] - 1
    # iou = tp/(tp + fp + fn)
    # the number of true positive pixels for this class
    # the entry on the diagonal of the confusion matrix
    tp = confusion_matrix[single_class, single_class]

    # the number of false negative pixels for this class
    # the column sum of the matching row in the confusion matrix
    # minus the diagonal entry
    fn = confusion_matrix[:, single_class].sum() - tp

    # the number of false positive pixels for this class
    # Only pixels that are not on a pixel with ground truth class that is ignored
    # The row sum of the corresponding row in the confusion matrix
    # without the ignored rows and without the actual label of interest
    not_ignored = [c for c in CLASS_INFO[experiment][1].keys() if not (c == 255 or c == single_class)]
    fp = confusion_matrix[single_class, not_ignored].sum()

    # the denominator of the IOU score
    denom = (tp + fp + fn)
    if denom == 0:
        # return float('nan')
        return 0  # Otherwise the mean always returns NaN which is technically correct but not so helpful
    # return IOU
    return float(tp) / denom
