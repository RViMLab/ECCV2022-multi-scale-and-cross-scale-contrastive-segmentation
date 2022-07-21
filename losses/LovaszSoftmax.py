import torch
import torch.nn as nn
from torch.nn.functional import softmax
from utils import DATASETS_INFO
from itertools import filterfalse


class LovaszSoftmax(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps = torch.as_tensor(1e-10)
        self.experiment = config['experiment']
        self.dataset = config['dataset']

        ignore_index_in_loss = len(
            DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]) - 1 \
            if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] else None
        # self.num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1])
        self.per_image = False if 'per_image' not in config else config['per_image']
        self.classes_to_ignore = ignore_index_in_loss if 'classes_to_ignore' not in config else config['classes_to_ignore']
        self.classes_to_consider = 'present' if 'classes_to_consider' not in config else config['classes_to_consider']
        # classes_to_consider: 'all' for all, 'present' for classes present in labels, or a list of classes to average

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-class Lovasz-Softmax loss. Adapted from github.com/bermanmaxim/LovaszSoftmax

        :param prediction: NCHW tensor, raw logits from the network
        :param target: NHW tensor, ground truth labels
        :return: Lovász-Softmax loss
        """
        p = softmax(prediction, dim=1)
        if self.per_image:
            loss = mean(self.lovasz_softmax_flat(*self.flatten_probabilities(p.unsqueeze(0), t.unsqueeze(0)))
                        for p, t in zip(p, target))
        else:
            loss = self.lovasz_softmax_flat(*self.flatten_probabilities(p, target))
        return loss

    def lovasz_softmax_flat(self, prob: torch.Tensor, lbl: torch.Tensor) -> torch.Tensor:
        """Multi-class Lovasz-Softmax loss. Adapted from github.com/bermanmaxim/LovaszSoftmax

        :param prob: class probabilities at each prediction (between 0 and 1)
        :param lbl: ground truth labels (between 0 and C - 1)
        :return: Lovász-Softmax loss
        """
        if prob.numel() == 0:
            # only void pixels, the gradients should be 0
            return prob * 0.
        c = prob.size(1)
        losses = []
        class_to_sum = list(range(c)) if self.classes_to_consider in ['all', 'present'] else self.classes_to_consider

        if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] and c in class_to_sum:
            class_to_sum.remove(c)  # remove ignore class which is denoted in lbl by values = c

        for c in class_to_sum:
            fg = (lbl == c).float()  # foreground for class c
            if self.classes_to_consider is 'present' and fg.sum() == 0:
                continue
            class_pred = prob[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.detach()
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
        return mean(losses)

    def flatten_probabilities(self, prob: torch.Tensor, lbl: torch.Tensor):
        """
        Flattens predictions in the batch
        """
        if prob.dim() == 3:
            # assumes output of a sigmoid layer
            n, h, w = prob.size()
            prob = prob.view(n, 1, h, w)
        _, c, _, _ = prob.size()
        prob = prob.permute(0, 2, 3, 1).contiguous().view(-1, c)  # B * H * W, C = P, C
        lbl = lbl.view(-1)
        if self.classes_to_ignore is None:
            return prob, lbl
        else:
            valid = (lbl != self.classes_to_ignore)
            vprobas = prob[valid.nonzero().squeeze()]
            vlabels = lbl[valid]
            return vprobas, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(ip: torch.Tensor, ignore_nan: bool = False, empty=0):
    """
    nanmean compatible with generators.
    """
    ip = iter(ip)
    if ignore_nan:
        ip = filterfalse(isnan, ip)
    try:
        n = 1
        acc = next(ip)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(ip, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
