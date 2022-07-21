import numpy as np
import pandas as pd
import torch
from torch.nn import Conv2d, ConvTranspose2d, ReLU, Sequential, BatchNorm2d
from torch.nn.functional import one_hot, unfold, pad
from torch.utils.data import Sampler
from utils import softmax, DATASETS_INFO, DEFAULT_VALUES
from torchvision.transforms import ToTensor
from typing import Union


class Compose2(object):
    """ torchvision Compose but __call__ can receive two arguments -- only for cityscapes compatible dataset transforms
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *img_lbl_tuple) -> tuple:
        for t in self.transforms:
            img_lbl_tuple = t(img_lbl_tuple)
        return img_lbl_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor2(object):
    def __init__(self):
        self.t = ToTensor()

    def __call__(self, img_lbl_tuple) -> tuple:
        img = self.t(img_lbl_tuple[0])
        return img, img_lbl_tuple[1]


def clipped_argmax(softmax_pred: torch.Tensor, t: float, ignore_value: int):
    """ given N,C,H,W softmax_pred tensor and a threshold value t
        set pixels above it to argmax(dim=1) else set ignore_value
     :return lbl of shape N,H,W
     """
    assert (0 <= t < 1), 'threshold must be in [0,1) instead got {}'.format(t)
    assert (ignore_value)
    scores, indices = torch.max(softmax_pred, dim=1)
    ignore = torch.tensor(ignore_value, dtype=indices.dtype)
    ignore_tensor = ignore * torch.ones(size=scores.size(), dtype=indices.dtype)
    ignore_tensor = ignore_tensor.to(softmax_pred.device)
    # only set as label the argmax of scores if max score is
    # above threshold t else set as label the ignore class
    lbl = torch.where(scores < t, ignore_tensor, indices)
    return lbl


def to_one_hot(tensor: torch.Tensor, n_classes: int) -> torch.Tensor:
    """One-hot representation of a NHW label tensor, outputs NCHW (therefore better than nn.functional.one_hot)"""
    n, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, h, w).to(tensor.device).scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot


def padded_conv2d(in_channels, out_channels, kernel_size,
                  stride=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
    # o = output
    # p = padding
    # k = kernel_size
    # s = stride
    # d = dilation
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - stride + 1) // 2  # +1 so // turns into ceil
    layer = Conv2d(in_channels=in_channels,
                   out_channels=out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   groups=groups,
                   bias=bias,
                   padding_mode=padding_mode)
    return layer


def padded_convtranspose2d(in_channels, out_channels, kernel_size, output_padding=0,
                           stride=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    # o = (i - 1) * s - 2 * input_padding + k + output_padding
    # o = output
    # k = kernel_size
    # s = stride
    padding = (kernel_size - stride + output_padding + 1) // 2  # +1 so // turns into ceil
    layer = ConvTranspose2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            output_padding=output_padding,
                            dilation=dilation,
                            groups=groups,
                            bias=bias,
                            padding_mode=padding_mode)
    return layer


def conv3x3(in_planes: int, out_planes: int, batch_norm: bool, relu: bool, stride: int = 1):
    """3x3 convolution with padding: https://github.com/CSAILVision/semantic-segmentation-pytorch"""
    c = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    b, r = None, None
    if batch_norm:
        b = BatchNorm2d(out_planes)
    if relu:
        r = ReLU(inplace=True)
    if not batch_norm and not relu:
        return c
    if batch_norm and not relu:
        return Sequential(c, b)
    if not batch_norm and relu:
        return Sequential(c, r)
    if batch_norm and relu:
        return Sequential(c, b, r)


def sliding_miou(prediction: torch.Tensor, target: torch.Tensor,
                 kernel_size: int, stride: int, original_size: bool = True) -> torch.Tensor:
    assert(kernel_size % 2 == 1), "Kernel size needs to be odd"
    with torch.no_grad():
        n, num_classes, h, w = prediction.shape
        p = prediction.argmax(1)                                                    # p is [N, H, W]
        p = to_one_hot(p, num_classes)                                              # p is [N, C, H, W]
        p = unfold(p, kernel_size=kernel_size, stride=stride).to(torch.int).view(n, num_classes, kernel_size**2, -1)
        # p is [N, C, win size, num wins]
        t = to_one_hot(target, num_classes)                                         # t is [N, C, H, W]
        t = unfold(t, kernel_size=kernel_size, stride=stride).to(torch.int).view(n, num_classes, kernel_size**2, -1)
        # t is [N, C, win size, num wins]
        intersection = torch.sum(p & t, dim=2).to(torch.float)                      # intersection is [N, C, num wins]
        union = torch.sum(p | t, dim=2).to(torch.float)                             # union is [N, C, num wins]
        i_over_u = intersection / union                                             # i_over_u is [N, C, num wins]
        i_over_u[union == 0] = 1
        # i_over_u[torch.isinf(i_over_u)] = 1  # alternative
        m_i_over_u = torch.mean(i_over_u.float(), dim=1)                            # m_i_over_u is [N, num wins]
        ver_num_wins = int(((h - kernel_size) / stride) // 1) + 1
        hor_num_wins = int(((w - kernel_size) / stride) // 1) + 1
        m_i_over_u = m_i_over_u.view((n, ver_num_wins, hor_num_wins))               # m_i_over_u is [N, v num, h num]
        if original_size:
            m_i_over_u = torch.repeat_interleave(torch.repeat_interleave(m_i_over_u, stride, dim=-2), stride, dim=-1)
            offset = kernel_size // 2
            offset_bottom = prediction.shape[-2] - m_i_over_u.shape[-2] - offset  # just in case
            offset_right = prediction.shape[-1] - m_i_over_u.shape[-1] - offset
            m_iou_tensor = pad(m_i_over_u, (offset, offset_right, offset, offset_bottom))  # m_iou_tensor is [N, H, W]
            return m_iou_tensor
        else:
            return m_i_over_u


def t_get_confusion_matrix(prediction: torch.Tensor, target: torch.Tensor, dataset: str,
                           existing_matrix: Union[None, torch.Tensor] = None, no_ignore_class: bool = True):
    """Expects prediction logits (as output by network), and target as classes in single channel (as from data)"""
    """no ignore class set to True --> no ignore class considered in the confusion matrix"""
    with torch.no_grad():
        num_classes = prediction.shape[1]  # prediction is shape NCHW, we want C (one-hot length of all classes)
        p = prediction.transpose(1, 0)  # Prediction is NCHW -> move C to the front to make it CNHW
        p = p.contiguous().view(num_classes, -1)  # Prediction is [C, N*H*W]
        p = p.argmax(0)  # Prediction is now [N*H*W]
        one_hot_pred = one_hot(p, num_classes).transpose(1, 0)  # adding +1 for the ignore class
        t = target.view(-1).to(torch.int64)

        list_of_num_classes_of_experiments_with_ignore_class = [len(c_info[1]) -1 for c_info
                                                                in DATASETS_INFO[dataset].CLASS_INFO if 255 in c_info[1]]

        if no_ignore_class and num_classes in list_of_num_classes_of_experiments_with_ignore_class:
            # for Experiments with ignore class : ommit the last ('ignore') class from metric
            one_hot_target = one_hot(t, num_classes + 1)
            # num_classes infered by prediction so has one less class than target --> i.e num_classes + 1 = all classes
            one_hot_target = one_hot_target[:, :-1]
        else:
            one_hot_target = one_hot(t, num_classes)
        confusion_matrix = torch.matmul(one_hot_pred.to(torch.float), one_hot_target.to(torch.float)).to(torch.int)
        # [C, N*H*W] x [N*H*W, C] = [C, C]
        if existing_matrix is not None:
            confusion_matrix += existing_matrix
        return confusion_matrix


def t_normalise_confusion_matrix(matrix: torch.Tensor, mode: str):
    with torch.no_grad():
        if mode == 'row':
            row_sums = torch.sum(matrix, dim=1, dtype=torch.float)
            row_sums[row_sums == 0] = 1  # to avoid division by 0. Safe, because if sum = 0, all elements are 0 too
            norm_matrix = matrix.to(torch.float) / row_sums.unsqueeze(1)
        elif mode == 'col':
            col_sums = torch.sum(matrix, dim=0, dtype=torch.float)
            col_sums[col_sums == 0] = 1  # to avoid division by 0. Safe, because if sum = 0, all elements are 0 too
            norm_matrix = matrix.to(torch.float) / col_sums.unsqueeze(0)
        else:
            raise ValueError("Normalise confusion matrix: mode needs to be either 'row' or 'col'.")
        return norm_matrix


def t_get_pixel_accuracy(confusion_matrix: torch.Tensor):
    """Pixel accuracies, adapted from https://github.com/CSAILVision/semantic-segmentation-pytorch

    :param confusion_matrix: Confusion matrix with absolute values. Rows are predicted classes, columns ground truths
    :return: Overall pixel accuracy, pixel accuracy per class (PA / PAC in CaDISv2 paper)
    """
    with torch.no_grad():
        pred_class_correct = torch.diag(confusion_matrix).to(torch.float)
        acc = torch.sum(pred_class_correct) / torch.sum(confusion_matrix)
        pred_class_sums = torch.sum(confusion_matrix, dim=1, dtype=torch.float)
        pred_class_sums[pred_class_sums == 0] = 1  # To avoid div by 0 problems. Safe because all elem = 0 when sum = 0
        acc_per_class = torch.mean(pred_class_correct / pred_class_sums)
        return acc, acc_per_class


def t_get_mean_iou(confusion_matrix: torch.Tensor, experiment: int, dataset: str,
                   categories=False, single_class=None, calculate_mean=None, rare=False):
    calculate_mean = True if calculate_mean is None else calculate_mean
    assert experiment in [1, 2, 3], 'experiment must be in [1,2,3] instead got [{}]'.format(experiment)
    mious = dict()
    if single_class is not None:
        # compute miou for a single_class
        assert(not categories),\
            'when single_class is not None, category must be False instead got [{}]'.format(categories)
        assert(single_class in DATASETS_INFO[dataset].CLASS_INFO[experiment]),\
            'single_class must be {} instead got [{}]'.format(DATASETS_INFO[dataset].CLASS_INFO[experiment][1].keys(), single_class)
        return t_get_single_class_iou(confusion_matrix, experiment, single_class)
    elif categories:
        # compute miou for all classes
        # compute miou for the classes of instruments and for the classes of anatomies
        assert (single_class is None),\
            'when category is not None, single class must be None instead got [{}]'.format(single_class)
        # compute miou for all classes
        mious['mean_iou'] = t_get_miou(confusion_matrix, experiment, dataset, calculate_mean=calculate_mean)
        mious['per_class_iou'] = t_get_miou(confusion_matrix, experiment, dataset, calculate_mean=False)

        # category iou
        mious['categories'] = dict()
        for categ in DATASETS_INFO[dataset].CLASS_INFO[experiment][2]:  # iterate over categories for given experiment
            mious['categories'][categ] = t_get_miou(confusion_matrix, experiment,
                                                    indices=DATASETS_INFO[dataset].CLASS_INFO[experiment][2][categ],
                                                    calculate_mean=calculate_mean, dataset=dataset)
        if rare and 'rare' in DATASETS_INFO[dataset].CLASS_INFO[experiment][2]:
            # todo add rare categories in utils categories_{experiment}
            # see utils categories categories_exp[X] for list of rare classes
            mious['categories']['rare'] = t_get_miou(confusion_matrix, experiment,
                                                     indices=DATASETS_INFO[dataset].CLASS_INFO[experiment][2]['rare'],
                                                     calculate_mean=calculate_mean, dataset=dataset)
            return mious
        else:
            return mious
    else:
        # compute miou for all classes
        mious['mean_iou'] = t_get_miou(confusion_matrix, experiment, dataset, calculate_mean=calculate_mean)
        return mious


def t_get_miou(confusion_matrix: torch.Tensor, experiment: int, dataset: str, indices=None, calculate_mean: bool = None):
    calculate_mean = True if calculate_mean is None else calculate_mean
    if indices is None:
        # all but the ignored indices
        indices = [c for c in DATASETS_INFO[dataset].CLASS_INFO[experiment][1].keys() if not c == 255]
    else:
        # indices can only be any of the categories of a given experiment
        # assert (indices == DATASETS_INFO[dataset].CLASS_INFO[experiment][2]['anatomies'] or
        #         indices == DATASETS_INFO[dataset].CLASS_INFO[experiment][2]['instruments'] or
        #         indices == DATASETS_INFO[dataset].CLASS_INFO[experiment][2]['rare'] or
        #         indices == DATASETS_INFO[dataset].CLASS_INFO[experiment][2]['others']),\
        #     'indices must be any of the entries of {}'.format(DATASETS_INFO[dataset].CLASS_INFO[experiment][2])
        indices = [c for c in indices if not c == 255]

    with torch.no_grad():
        diagonal = confusion_matrix.diag()[indices].to(torch.float)
        row_sum = torch.sum(confusion_matrix, dim=0, dtype=torch.float)[indices]
        col_sum = torch.sum(confusion_matrix, dim=1, dtype=torch.float)[indices]
        denominator = row_sum + col_sum - diagonal
        iou = diagonal / denominator
        iou[iou != iou] = 0  # if iou of some class is Nan (i.e denominator was 0) set it to 0 to avoid Nan in the mean
        if calculate_mean:
            mean_iou = iou.mean()
            return mean_iou
        else:
            return iou


def t_get_single_class_iou(confusion_matrix: torch.Tensor, experiment: int, single_class: int, dataset: str):
    with torch.no_grad():
        if single_class == 255:
            single_class = confusion_matrix.shape[0] - 1
        indices = [c for c in DATASETS_INFO[dataset].CLASS_INFO[experiment][1].keys() if not (c == 255 or c == single_class)]
        tp = confusion_matrix[single_class, single_class]
        fn = torch.sum(confusion_matrix[:, single_class]) - tp
        fp = torch.sum(confusion_matrix[single_class, indices])
        denom = tp + fp + fn
        if denom.cpu().numpy() == 0:
            return torch.zeros(1)
        return tp.to(torch.float) / denom.to(torch.float)
