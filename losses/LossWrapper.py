import torch
from torch import nn
# noinspection PyUnresolvedReferences
from losses import *
from utils import DATASETS_INFO
from typing import Union


class LossWrapper(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.loss_weightings = config['losses']
        self.device = config['device']
        self.dataset = config['dataset']
        self.experiment = config['experiment']
        self.total_loss = None
        self.loss_classes, self.loss_vals = {}, {}
        self.info_string = ''
        self.ignore_class = (len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]) - 1) \
            if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] else -1
        for loss_class in self.loss_weightings:
            if loss_class == 'CrossEntropyLoss':
                class_weights = None
                if self.dataset == 'CITYSCAPES':
                    class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                                                       0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                                       1.0865, 1.1529, 1.0507]).cuda()
                print(f'using class_weights {class_weights}')
                loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_class, weight=class_weights)
                # loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_class)
            else:
                loss_fct = globals()[loss_class](config)
            self.loss_classes.update({loss_class: loss_fct})
            self.loss_vals.update({loss_class: 0})
            self.info_string += loss_class + ', '
        self.info_string = self.info_string[:-2]
        self.dc_off = True if 'dc_off_at_epoch' in self.config else False

    def forward(self,
                prediction: torch.Tensor,
                labels: torch.Tensor,
                loss_list: list = None,
                deep_features: Union[torch.Tensor,list] = None,
                interm_prediction: torch.Tensor = None,
                epoch: int = None,
                skip_mem_update: bool =False) -> torch.Tensor:
        self.total_loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
        # Compile list of losses to be evaluated. If no specific 'loss_list' is passed
        loss_list = list(self.loss_weightings.keys()) if loss_list is None else loss_list
        for loss_class in self.loss_weightings:  # Go through all the losses
            if loss_class in loss_list:  # Check if this loss should be calculated
                if 'DenseContrastive' in loss_class:
                    assert deep_features is not None, f'for loss_class {loss_class}, deep_features must be tensor (B,H,W,C) ' \
                                                      f'instead got {deep_features}'
                if loss_class == 'LovaszSoftmax':
                    if self.dc_off and epoch is not None and epoch < self.config['dc_off_at_epoch']:
                        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
                    else:
                        loss = self.loss_classes[loss_class](prediction, labels)
                elif loss_class == 'DenseContrastiveLoss':
                    if self.dc_off and epoch is not None and epoch >= self.config['dc_off_at_epoch']:
                        loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
                    else:
                        loss = self.loss_classes[loss_class](labels, deep_features)
                elif loss_class == 'TwoScaleLoss':
                    loss = self.loss_classes[loss_class](interm_prediction, prediction, labels.long())
                elif loss_class == 'DenseContrastiveLossV2':
                    loss = self.loss_classes[loss_class](labels, deep_features)
                elif loss_class == 'DenseContrastiveLossV2_ms':
                    loss = self.loss_classes[loss_class](labels, deep_features)
                elif loss_class == 'DenseContrastiveLossV3':
                    loss = self.loss_classes[loss_class](labels, deep_features, epoch, skip_mem_update)
                elif loss_class == 'DenseContrastiveLossV3_ms':
                    loss = self.loss_classes[loss_class](labels, deep_features, epoch, skip_mem_update)
                    # self.meta['queue'] = self.loss_classes[loss_class].queue_ptr.clone().numpy()
                elif loss_class == 'DenseContrastiveCenters':
                    loss = self.loss_classes[loss_class](labels, deep_features, epoch, skip_mem_update)
                elif loss_class == 'OhemCrossEntropy':
                    loss = self.loss_classes[loss_class](prediction, labels)
                elif loss_class == 'CrossEntropyLoss':
                    loss = self.loss_classes[loss_class](prediction, labels)
                else:
                    print("Error: Loss class '{}' not recognised!".format(loss_class))
                    loss = torch.tensor(0.0, dtype=torch.float, device=self.device)
            else:
                loss = torch.tensor(0.0, dtype=torch.float, device=self.device)

            # Calculate weighted loss
            loss *= self.loss_weightings[loss_class]
            self.loss_vals[loss_class] = loss.detach()

            # logging each scale seperately if ms/cs loss
            if loss_class == 'DenseContrastiveLossV2_ms':

                if hasattr(self.loss_classes[loss_class], 'ms_losses'):
                    for scale, loss_val_ms in enumerate(self.loss_classes[loss_class].ms_losses):
                        self.loss_vals.update({f'{loss_class}_ms{scale}':loss_val_ms})
                if self.loss_classes[loss_class].cross_scale_contrast and hasattr(self.loss_classes[loss_class], 'cs_losses'):
                    for cscale, loss_val_cs in enumerate(self.loss_classes[loss_class].cs_losses):
                        self.loss_vals.update({f'{loss_class}_cs{cscale}':loss_val_cs})
            self.total_loss += loss
        return self.total_loss
