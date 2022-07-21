from managers.BaseManager import BaseManager
from utils import to_comb_image, t_get_confusion_matrix, t_normalise_confusion_matrix, t_get_pixel_accuracy, \
    get_matrix_fig, to_numpy, t_get_mean_iou, DATASETS_INFO, printlog
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import datetime
from models import HRNet
from losses import LossWrapper
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm



class HRNetManager(BaseManager):

    def forward_step(self, img, lbl, **kwrargs):
        ret = dict()

        skip_mem_update = False
        if 'skip_mem_update' in kwrargs:
            skip_mem_update = kwrargs['skip_mem_update']

        if isinstance(self.loss, LossWrapper):
            if self.return_features:
                output, proj_features = self.model(img.float())
                loss = self.loss(output, lbl.long(), deep_features=proj_features, epoch=self.epoch, skip_mem_update=skip_mem_update)
            else:
                output = self.model(img.float())
                proj_features = None
                loss = self.loss(output, lbl.long(), epoch=self.epoch)

            # get individual loss terms values for logging
            if 'individual_losses' in kwrargs:
                individual_losses = kwrargs['individual_losses']
                for key in self.loss.loss_vals:
                    individual_losses[key] += self.loss.loss_vals[key]
                ret['individual_losses'] = individual_losses

        else:
            # not using the LossWrapper module
            output = self.model(img.float())
            proj_features = None
            loss = self.loss(output, lbl.long())

        ret['output'] = output
        ret['interm_output'] = None
        ret['feats'] = proj_features
        ret['loss'] = loss

        if self.empty_cache:
            torch.cuda.empty_cache()
        return ret

    def post_process_output(self, img, output, lbl, metadata, skip_label=False):
        if metadata and self.dataset in ['PASCALC', 'ADE20K']:
            if "pw_ph_stride" in metadata:
                # undo padding due to fit_stride resizing
                pad_w, pad_h, stride = metadata["pw_ph_stride"]
                if pad_h > 0 or pad_w > 0:
                    output = output[:, :, 0:output.size(2) - pad_h, 0:output.size(3) - pad_w]
                    lbl = lbl[:, 0:output.size(2) - pad_h, 0:output.size(3) - pad_w]
                    img = img[:, :, 0:output.size(2) - pad_h, 0:output.size(3) - pad_w]

            if "sh_sw_in_out" in metadata:
                if hasattr(self.model, 'module'):
                    align_corners = self.model.module.align_corners
                else:
                    align_corners = self.model.align_corners
                # undo resizing
                starting_size = metadata["sh_sw_in_out"][-2]
                # starting size is w,h due to fucking PIL
                output = F.interpolate(input=output, size=starting_size[-2:][::-1],
                                       mode='bilinear', align_corners=align_corners)
                img = F.interpolate(input=img, size=starting_size[-2:][::-1],
                                    mode='bilinear', align_corners=align_corners)
                lbl = metadata["original_labels"].squeeze(0).long().cuda()

        return img, output, lbl

    def train_one_epoch(self):
        """Train the model for one epoch"""
        if self.rank == 0 and self.epoch == 0 and self.parallel:
           printlog('worker rank {} : CREATING train_writer'.format(self.rank))
           self.train_writer = SummaryWriter(log_dir = self.log_dir / 'train')

        self.model.train()
        a = datetime.datetime.now()
        running_confusion_matrix = 0
        for batch_num, batch in enumerate(self.data_loaders[self.train_schedule[self.epoch]]):
            if len(batch) == 2:
                img, lbl = batch
            else:
                img, lbl, metadata = batch
            # if self.debugging:
            #     continue
            b = (datetime.datetime.now() - a).total_seconds() * 1000
            a = datetime.datetime.now()
            img, lbl = img.to(self.device, non_blocking=True), lbl.to(self.device, non_blocking=True)
            self.optimiser.zero_grad()
            # forward
            ret = self.forward_step(img, lbl)
            loss = ret['loss']
            output = ret['output']
            # backward
            loss.backward()
            self.optimiser.step()
            # lr scheduler
            if self.scheduler is not None and self.config['train']['lr_batchwise']:
                self.scheduler.step()

            if batch_num == 2 and self.debugging:
               break

            # logging
            confusion_matrix = t_get_confusion_matrix(output, lbl, self.dataset)
            running_confusion_matrix += confusion_matrix
            pa, pac = t_get_pixel_accuracy(confusion_matrix)
            mious = t_get_mean_iou(confusion_matrix, self.config['data']['experiment'],
                                   self.dataset, categories=True, calculate_mean=False, rare=True)
            self.train_logging(batch_num, output, img, lbl, mious, loss, pa, pac, b)

        if 'DenseContrastiveLoss' in self.loss.loss_classes:
            col_confusion_matrix = t_normalise_confusion_matrix(running_confusion_matrix, mode='col')
            self.train_writer.add_figure('train_confusion_matrix/col_normalised',
                                         get_matrix_fig(to_numpy(col_confusion_matrix),
                                                        self.config['data']['experiment'],
                                                        self.dataset), self.global_step - 1)
            self.loss.loss_classes['DenseContrastiveLoss'].update_confusion_matrix(col_confusion_matrix)

        meta = {}
        if 'DenseContrastiveLossV3' in self.loss.loss_classes:
            meta = self.loss.loss_classes['DenseContrastiveLossV3'].get_meta()
        elif 'DenseContrastiveCenters' in self.loss.loss_classes:
            meta = self.loss.loss_classes['DenseContrastiveCenters'].get_meta()

        if 'queue_fillings' in meta:
            # self.num_real_classes, dtype=torch.long
            self.config['queue_fillings'] = meta['queue_fillings']
            self.write_info_json()

        if self.scheduler is not None and not self.config['train']['lr_batchwise']:
            self.scheduler.step()
            self.train_writer.add_scalar('parameters/learning_rate', self.scheduler.get_lr()[0], self.global_step) \
                if self.rank == 0 else None

    def validate(self):
        """Validate the model on the validation data"""
        if self.rank == 0:
            # only process with rank 0 runs validation step
            if self.epoch == 0 and self.parallel:
                printlog(f'\n creating valid_writer ... for process rank {self.rank}')
                self.valid_writer = SummaryWriter(log_dir= self.log_dir / 'valid')
        else:
            return

        if not self.parallel:
            torch.backends.cudnn.benchmark = False

        self.model.eval()
        valid_loss = 0
        confusion_matrix = None
        individual_losses = dict()
        if isinstance(self.loss, LossWrapper):
            for key in self.loss.loss_vals:
                individual_losses[key] = 0
            if 'DenseContrastiveLossV3' in self.loss.loss_classes: # make loss run the non ddp version for validation
                self.loss.loss_classes['DenseContrastiveLossV3'].parallel = False

        with torch.no_grad():
            for rec_num, batch in enumerate(tqdm(self.data_loaders['valid_loader'])):
                if len(batch) == 2:
                    img, lbl = batch
                    metadata = None
                else:
                    img, lbl, metadata = batch
                img, lbl = img.to(self.device, non_blocking=True), lbl.to(self.device, non_blocking=True)

                # forward
                ret = self.forward_step(img, lbl, individual_losses=individual_losses, skip_mem_update=True)
                loss = ret['loss']
                output = ret['output']
                valid_loss += loss
                img, output, lbl = self.post_process_output(img, output, lbl, metadata)

                # logging
                confusion_matrix = t_get_confusion_matrix(output, lbl, self.dataset, confusion_matrix)
                if rec_num in np.round(np.linspace(0, len(self.data_loaders['valid_loader']) - 1, self.max_valid_imgs)):
                    lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                    self.valid_writer.add_image(
                        'valid_images/record_{:02d}'.format(rec_num),
                        to_comb_image(self.un_norm(img)[0], lbl[0], lbl_pred[0], self.config['data']['experiment'], self.dataset),
                        self.global_step, dataformats='HWC')
                individual_losses= ret['individual_losses'] if 'individual_losses' in ret else individual_losses
                if self.debugging and rec_num == 2:
                    break
        valid_loss /= len(self.data_loaders['valid_loader'])
        pa, pac = t_get_pixel_accuracy(confusion_matrix)
        mious = t_get_mean_iou(confusion_matrix, self.config['data']['experiment'], self.dataset, True, rare=True)
        # logging + checkpoint
        self.valid_logging(valid_loss, confusion_matrix, individual_losses, mious, pa, pac)

        if not self.parallel:
            torch.backends.cudnn.benchmark = True

        if isinstance(self.loss, LossWrapper):
            if 'DenseContrastiveLossV3' in self.loss.loss_classes: # reset
               self.loss.loss_classes['DenseContrastiveLossV3'].parallel = self.parallel
