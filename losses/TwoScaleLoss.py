import torch.nn.functional as F
import torch.nn as nn
from utils import DATASETS_INFO
from losses import LovaszSoftmax
from torch.nn import CrossEntropyLoss
import torch


class TwoScaleLoss(nn.Module):
    def __init__(self, config):
        """
         Loads two losses one from an intermediate output and one from the final output
         for now it assumes the two losses are the same CE-CE or Lovasz-Lovasz etc.
         the weights of the two losses may vary (by default 0.4 for interm and 1.0 final)
        :param config:
        """
        super(TwoScaleLoss, self).__init__()
        interm_loss_class = globals()[config['interm']['name']]
        final_loss_class = globals()[config['final']['name']]
        self.w_interm = config['interm']['weight'] if 'weight' in config['interm'] else 0.4
        self.w_final = config['final']['weight'] if 'weight' in config['final'] else 1.0
        self.ignore_label = -100  # if experiment is not given assume nothing is ignored
        self.dataset = config['dataset']
        self.experiment = config['experiment']
        if 'experiment' in config:
            self.ignore_label = len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]) - 1 \
                if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1].keys() \
                else len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1])

        # pass experiment id to constructors of the two losses
        config['interm'].update({"experiment": config['experiment'], "dataset": self.dataset})
        config['final'].update({"experiment": config['experiment'], "dataset": self.dataset})

        if config['interm']['name'] == 'CrossEntropyLoss' and config['final']['name'] == 'CrossEntropyLoss':
            class_weights = None
            if self.dataset == 'CITYSCAPES':
                class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                                                   0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                                   1.0865, 1.1529, 1.0507]).cuda()
                print(f'using class weights {class_weights}')
            self.loss_interm = interm_loss_class(*config['interm']['args'],
                                                 ignore_index=self.ignore_label, weight=class_weights)
            self.loss_final = final_loss_class(*config['final']['args'],
                                               ignore_index=self.ignore_label, weight=class_weights)

            # all other losses expect a config
        elif config['interm']['name'] == config['final']['name']:
            self.loss_interm = interm_loss_class(config['interm'])
            self.loss_final = final_loss_class(config['final'])
        else:
            raise NotImplementedError('different losses for interm {}'
                                      ' and final {}'.format(config['interm'], config['final']))

        print("intermediate loss {} with weight {}".format(interm_loss_class, self.w_interm))
        print("final       loss {} with weight {}".format(final_loss_class, self.w_final))

    def forward(self, logits_interm, logits_final, target):
        # upsample intermediate if not already upsampled
        ph, pw = logits_interm.size(2), logits_interm.size(3)
        h, w = target.size(1), target.size(2)
        # todo add align_corners from outside --
        #  this was ignored until now as upsampling was happening in model.forward()
        # if ph != h or pw != w:
        #   logits_interm = F.upsample(input=logits_interm, size=(h, w), mode='bilinear')
        loss_final = self.loss_final(logits_final, target)
        loss_interm = self.loss_interm(logits_interm, target)
        loss = loss_final * self.w_final + loss_interm * self.w_interm
        return loss
