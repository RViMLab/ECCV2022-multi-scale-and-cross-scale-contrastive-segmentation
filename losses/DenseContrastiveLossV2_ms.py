import torch
import torch.nn as nn
from utils import DATASETS_INFO, is_distributed, concat_all_gather, get_rank, to_numpy, printlog
from torch.nn.functional import one_hot
import torch.distributed
import numpy as np
import datetime
from losses.DenseContrastiveLossV2 import DenseContrastiveLossV2 as DCV2

def has_inf_or_nan(x):
    return torch.isinf(x).max().item(), torch.isnan(x).max().item()
# -c configs/OCRNet_contrastive_CADIS.json  -u theo -cdn False  -debug

class DenseContrastiveLossV2_ms(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.parallel = is_distributed()
        self.experiment = config['experiment']
        self.dataset = config['dataset']
        self.num_all_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1])
        self.num_real_classes = self.num_all_classes - 1 if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] else self.num_all_classes
        self.ignore_class = (len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]) - 1) if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] else -1
        self.scales = config['scales'] if 'scales' in config else 2
        self.weights = config['weights'] if 'weights' in config else [1.0] * self.scales
        assert(self.scales == len(self.weights)), f'given dc loss number of scales [{self.scales}] not equal len of weights {self.weights}'
        self.losses = []
        self.eps = torch.tensor(1e-10)
        self.meta = {}
        self.cross_scale_contrast = config['cross_scale_contrast'] if 'cross_scale_contrast' in config else False
        self.cross_scale_temperature = config['temperature'] if 'cross_scale_temperature' not in config else 0.1
        self.detach_cs_deepest = config['detach_deepest'] if 'detach_deepest' in config else False
        self.w_high_low = config['w_high_low'] if 'w_high_low' in config else 1.0
        self.w_high_mid = config['w_high_mid'] if 'w_high_mid' in config else 1.0
        self.ms_losses = []
        self.cs_losses = []
        printlog(f'defining dcv2 ms loss with number of scales {self.scales} and weights {self.weights}')
        printlog(f'using cross scale contrast {self.cross_scale_contrast}')
        for class_name in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]:
            self.meta[class_name] = (0.0, 0.0)  # pos-neg per class
        for s in range(self.scales):
            printlog(f'defining dcv2 loss at scale {s}')
            setattr(self, f'DCV2_scale{s}', DCV2(config))
        if self.cross_scale_contrast:
            printlog(f'using cross-scale contrast with detach_cs_deepest set to {self.detach_cs_deepest}, w_high_low: {self.w_high_low}, w_high_mid: {self.w_high_mid}')

    def forward(self, label: torch.Tensor, features: list, **kwargs):
        self.cs_losses = []
        self.ms_losses = []
        flag_error = False
        loss = torch.tensor(0.0, dtype=torch.float, device=features[0].device)
        feats_ms = []
        labels_ms = []
        for s in range(self.scales):
            if self.cross_scale_contrast:
                loss_s, feats_s, labels_s, flag_error = getattr(self, f'DCV2_scale{s}')(label, features[s])
                loss+= self.weights[s] * loss_s
                feats_ms.append(feats_s)
                labels_ms.append(labels_s)
            else:
                loss_s=getattr(self, f'DCV2_scale{s}')(label, features[s])
                loss += self.weights[s] * loss_s
            self.ms_losses.append(loss_s.detach())

        if self.cross_scale_contrast and not flag_error:
            assert len(feats_ms) > 1
            assert len(labels_ms) > 1
            # highest res to lowest res contrast
            if self.detach_cs_deepest:
                loss_cross_scale = self.contrastive_loss(feats_ms[0], labels_ms[0], feats_ms[-1].detach(), labels_ms[-1])
            else:
                loss_cross_scale = self.contrastive_loss(feats_ms[0], labels_ms[0], feats_ms[-1], labels_ms[-1])
                self.cs_losses.append(loss_cross_scale.detach())

            loss += self.w_high_low * loss_cross_scale

            if len(feats_ms)>2: # hrnet : 4 , s4-s16 , s4-s32 dlv3 : 3 layer1(s4)-layer4(s8), layer1(s4)-layer3(s8)
                if self.detach_cs_deepest:
                    loss_cross_scale2 = self.contrastive_loss(feats_ms[0], labels_ms[0], feats_ms[-2].detach(), labels_ms[-2])
                else:
                    loss_cross_scale2 = self.contrastive_loss(feats_ms[0], labels_ms[0], feats_ms[-2], labels_ms[-2])
                loss += self.w_high_mid * loss_cross_scale2
                self.cs_losses.append(loss_cross_scale2.detach())

        return loss

    def contrastive_loss(self, feats1, labels1, feats2, labels2):
        """
        :param feats: T-C-V
                      T: classes in batch (with repetition), which can be thought of as the number of anchors
                      C: feature space dimensionality
                      V: views per class (i.e samples from each class),
                       which can be thought of as the number of views per anchor
        :param labels: T
        :return: loss
        """
        # prepare feats
        feats1 = torch.nn.functional.normalize(feats1, p=2, dim=1)  # L2 normalization
        feats1 = feats1.transpose(dim0=1, dim1=2)  # feats are T-V-C
        num_anchors, views_per_anchor, c = feats1.shape  # get T, V, C
        feats_flat1 = feats1.contiguous().view(-1, c)  # feats_flat is T*V-C

        labels1 = labels1.contiguous().view(-1, 1)  # labels are T-1
        labels1 = labels1.repeat(1, views_per_anchor)  # labels are T-V
        labels1 = labels1.view(-1, 1)  # labels are T*V-1

        feats2 = torch.nn.functional.normalize(feats2, p=2, dim=1)  # L2 normalization
        feats2 = feats2.transpose(dim0=1, dim1=2)  # feats are T-V-C
        num_anchors, views_per_anchor, c = feats2.shape  # get T, V, C
        feats_flat2 = feats2.contiguous().view(-1, c)  # feats_flat is T*V-C

        labels2 = labels2.contiguous().view(-1, 1)  # labels are T-1
        labels2 = labels2.repeat(1, views_per_anchor)  # labels are T-V
        labels2 = labels2.view(-1, 1)  # labels are T*V-1

        pos_mask, neg_mask = self.get_masks2(labels1, labels2)
        dot_product = torch.div(torch.matmul(feats_flat1, torch.transpose(feats_flat2, 0, 1)), self.cross_scale_temperature)
        loss2 = self.InfoNce_loss(pos_mask, neg_mask, dot_product)
        return loss2


    @staticmethod
    def get_masks2(lbl1, lbl2):
        """
        takes flattened labels and identifies pos/neg of each anchor
        :param labels: T*V-1
        :param num_anchors: T
        :param views_per_anchor: V
        :return: mask, pos_maks,
        """
        # extract mask indicating same class samples
        pos_mask = torch.eq(lbl1, torch.transpose(lbl2, 0, 1)).float()  # mask T-T  # indicator of positives
        pos_sums = pos_mask.sum(1)
        # zero_pos_rows = (pos_sums == 0).nonzero().squeeze()
        # ignore_mask = torch.ones(size=pos_mask.size())[zero_pos_rows]
        neg_mask = (1 - pos_mask)  # indicator of negatives
        return pos_mask, neg_mask

    def InfoNce_loss(self, pos, neg, dot):
        """
        :param pos: V*T-V*T
        :param neg: V*T-V*T
        :param dot: V*T-V*T
        :return:
        """
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = dot  # - logits_max.detach()

        neg_logits = torch.exp(logits) * neg
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        # print('exp_logits ', has_inf_or_nan(exp_logits))
        log_prob = logits - torch.log(exp_logits + neg_logits)
        # print('log_prob ', has_inf_or_nan(log_prob))
        pos_sums = pos.sum(1)
        ones = torch.ones(size=pos_sums.size())
        norm = torch.where(pos_sums > 0, pos_sums, ones.to(pos.device))
        mean_log_prob_pos = (pos * log_prob).sum(1) / norm   # normalize by positives
        # print('\npositives: {} \nnegatives {}'.format(pos.sum(1), neg.sum(1)))
        # print('mean_log_prob_pos ', has_inf_or_nan(mean_log_prob_pos))
        loss = - mean_log_prob_pos
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.mean()
        # print('loss.mean() ', has_inf_or_nan(loss))
        # print('loss {}'.format(loss))
        if has_inf_or_nan(loss)[0] or has_inf_or_nan(loss)[1]:
            print('\n inf found in loss with positives {} and Negatives {}'.format(pos.sum(1), neg.sum(1)))
        return loss

    def get_meta(self):
        meta = {}
        meta['queue_fillings']= to_numpy(self.queue_fillings)
        meta['scales']= int(self.scales)
        return meta
