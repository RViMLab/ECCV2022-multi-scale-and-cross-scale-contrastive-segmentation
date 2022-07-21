import pathlib
import cv2
import torch
import datetime
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch import nn
import torch.nn.functional as f
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.transforms import ToPILImage, ToTensor
from datasets import DatasetFromDF
from datasets import Cityscapes as cts
from datasets import PascalC as pc
from datasets import ADE20K as ade
from datasets import get_cadis_dataframes
from models import *
from losses import *
from utils import DATASETS_INFO, get_class_info, RepeatFactorSampler, LRFcts, mask_to_colormap, \
    get_remapped_colormap, t_get_mean_iou, t_get_confusion_matrix, to_numpy, worker_init_fn, printlog, remap_mask, \
    create_new_directory, reverse_mapping, un_normalise, do_nothing, parse_transform_lists, is_in, \
    get_param_groups_with_stage_wise_lr_decay, set_seeds, get_param_groups_using_keys
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import builtins
import torch.backends.cudnn as cudnn
from utils import Logger as Log
from typing import Union
from .LoggingManager import LoggingManager


class BaseManager(LoggingManager):
    """Base Manager class, from which all model specific managers inherit"""
    def __init__(self, configuration):
        super().__init__(configuration)
        if self.parallel and self.config['mode'] == 'training':
            if self.debugging:
                os.environ['NCCL_DEBUG'] = 'INFO'
            # torch.manual_seed(self.config['seed'])
            set_seeds(self.config['seed'])
            self.rank = 0
            mp.spawn(fn=self.distributed_train_worker, nprocs=self.n_gpus)
            return

        cudnn.benchmark = self.config['cudnn_benchmark'] if 'cudnn_benchmark' in self.config else True
        printlog(f'*** setting cudnn.benchmark to {cudnn.benchmark} ***')
        printlog(f"*** cudnn.enabled {cudnn.enabled}")
        printlog(f"*** cudnn.deterministic {cudnn.deterministic}")

        if 'load_checkpoint' in self.config:
            self.config['graph']['pretrained'] = False  # do not load imagenet weights when loading checkpoint
        # Load model into self.model
        self.load_model()

        if self.config['mode'] == 'training' and not self.parallel:

            # Load loss into self.loss
            self.load_loss()

            # Set manual seeds to make training repeatable
            torch.manual_seed(self.config['seed'])

            # Load the datasets if given
            self.load_data()

            # Optimiser
            self.load_optimiser()

            # flags
            cudnn.benchmark = self.config['cudnn_benchmark'] if 'cudnn_benchmark' in self.config else True
            printlog(f'*** setting cudnn.benchmark to {cudnn.benchmark} ***')

            if 'load_checkpoint' in self.config:

                if 'load_last' in self.config:
                    chkpt_type = 'last' if self.config['load_last'] else 'best'
                    self.load_checkpoint(chkpt_type)
                else:
                    self.load_checkpoint('best')

            # Tensorboard writers
            self.train_writer = SummaryWriter(log_dir=self.log_dir / 'train')
            self.valid_writer = SummaryWriter(log_dir=self.log_dir / 'valid')


        elif self.config['mode'] == 'demo_tsne':
            # this mode loads only the validation dataset of a split
            cudnn.benchmark = self.config['cudnn'] if 'cudnn' in self.config else True
            printlog(f'*** setting cudnn.benchmark {cudnn.benchmark} ***')
            # self.valid_writer = SummaryWriter(log_dir=self.log_dir / 'infer')
            torch.manual_seed(self.config['seed'])
            self.load_data()

        elif self.config['mode'] == 'inference':
            # this mode loads only the validation dataset of a split
            cudnn.benchmark = self.config['cudnn'] if 'cudnn' in self.config else True
            printlog(f'*** setting cudnn.benchmark {cudnn.benchmark} ***')
            self.valid_writer = SummaryWriter(log_dir=self.log_dir / 'infer')
            torch.manual_seed(self.config['seed'])
            self.load_data()


    def distributed_train_worker(self, gpu):
        Log.init(logfile_level="info",
                 stdout_level=None,
                 log_file=self.log_file,
                 rewrite=True)
        set_seeds(self.config['seed'])
        if self.rank == 0:
            printlog("Run ID: {}".format(self.run_id))

        self.device = torch.device(f'cuda:{self.allocated_devices[gpu]}')
        if self.device == 100:
            def print_pass(*args):
                # this can be used to supress printing by other than the rank=0 process in ddp -- not used curently
                pass
            builtins.print = print_pass
        torch.cuda.set_device(self.device)
        cudnn.benchmark = self.config['cudnn_benchmark'] if 'cudnn_benchmark' in self.config else True
        # cudnn.benchmark = self.config['cudnn_enabled'] if 'cudnn_enabled' in self.config else True
        # cudnn.benchmark = self.config['cudnn_determinstic'] if 'cudnn_deterministic' in self.config else True
        printlog(f'*** setting cudnn.benchmark to {cudnn.benchmark} ***') if self.rank==0 else None
        printlog(f"*** cudnn.enabled {cudnn.enabled}") if self.rank==0 else None
        printlog(f"*** cudnn.deterministic {cudnn.deterministic}") if self.rank==0 else None
        self.batch_size = int(self.batch_size) // self.n_gpus
        # self.config['data']['num_workers'] = int((self.config['data']['num_workers'] + self.n_gpus - 1) / self.n_gpus)
        self.rank = self.rank * self.n_gpus + gpu
        printlog(f'Use GPU: {self.device} Rank: {self.rank}')
        dist.init_process_group(backend='nccl',
                                world_size=self.world_size,
                                rank=self.rank)
        self.load_model()
        self.load_loss()
        self.load_data()
        self.load_optimiser()
        if 'load_checkpoint' in self.config:
            if 'load_last' in self.config:
                chkpt_type = 'last' if self.config['load_last'] else 'best'
                self.load_checkpoint(chkpt_type)
            else:
                self.load_checkpoint('best')
        self.train()

    def train(self):
        """Main training loop"""
        printlog("***** Training started *****\n")
        for self.epoch in range(self.config['train']['epochs']):
            if (self.epoch + self.start_epoch) == int(0.9 * self.config['train']['epochs']):
                self.valid_freq = 1
                if self.rank == 0:
                    printlog(f'** train: validation to be run after every {self.valid_freq} epoch from now on')

            if self.parallel:
                self.train_sampler.set_epoch(self.epoch+self.start_epoch)

            if self.epoch == 0:
                t1 = datetime.datetime.now()

            self.train_one_epoch()

            if self.epoch == 0:
                t = (datetime.datetime.now() - t1).total_seconds()
                printlog('** Aprox. run t: {:.1f} total /{:.2f} per epoch'.format(t * (self.config['train']['epochs']-self.start_epoch) / 3600,
                                                                                  t / 3600 ))

            if (self.epoch + self.start_epoch) % self.valid_freq == 0:
                self.validate()

            elif (self.epoch + self.start_epoch-1) == self.config['train']['epochs']-1:
                self.validate()
                break

        if not self.rank == 0:  # stop
            return
        printlog("\n***** Training finished *****\n"
                 "Run ID: {}\n"
                 "     Best validation loss: {:.5f}\n".format(self.run_id, self.best_loss))
        msg_stra = "     Best mIoU        (tot"
        msg_strb = "     best loss mIoU   (tot "
        msg_strC = "     FINAL mIoU   (tot "
        msg1, msg2 = "", "{:.4f} ".format(self.metrics['best_miou'])
        msg3, msg4 = "", "{:.4f} ".format(self.metrics['best_loss_miou'])
        msg5, msg6 = "", "{:.4f} ".format(self.metrics['final_miou'])

        for categ in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][2]:
            msg1 += "/ {} ".format(categ)
            msg2 += "{:.4f} ".format(self.metrics['best_miou_{}'.format(categ)])
            msg3 += "/ {} ".format(categ)
            msg4 += "{:.4f} ".format(self.metrics['best_loss_miou_{}'.format(categ)])
            msg5 += "/ {} ".format(categ)
            msg6 += "{:.4f} ".format(self.metrics['final_miou_{}'.format(categ)])
        printlog(msg_stra + msg1 + ' ): ' + msg2 + '@ epoch {} (step {})'.format(*self.metrics['best_miou_epoch_step']))
        printlog(msg_strb + msg3 + ' ): ' + msg4 + '@ epoch {} (step {})'.format(*self.metrics['best_loss_epoch_step']))
        printlog(msg_strC + msg5 + ' ): ' + msg6 + '@ epoch {} (step {})'.format(self.config['train']['epochs'], self.global_step))

        self.finalise()
        if self.run_final_val and self.rank==0: # only rank=0 process runs this if ddp
            printlog(f'starting validation with final model r {self.rank}')
            self.config.update({"load_checkpoint": self.run_id, "load_last": True, "tta": True})
            self.infer()

    def load_data(self):
        if self.config['mode'] == 'inference':
            train_df, valid_df = self.get_seg_dataframes()
            _, valid_loader = self.get_dataloaders(train_df, valid_df, 'default')
            self.data_loaders = {'valid_loader': valid_loader}
            return
        train_df, valid_df = self.get_seg_dataframes()
        train_loader, valid_loader = self.get_dataloaders(train_df, valid_df, 'default')
        self.data_loaders = {'train_loader': train_loader,
                             'valid_loader': valid_loader}
        self.batches_per_epoch = len(train_loader)

        # Obtain schedule
        loader_type_list = ['repeat_factor']
        l_list = [self.config['data'][loader_type][0] for loader_type in loader_type_list]
        idx = np.argsort(l_list)
        for loader_type in np.array(loader_type_list)[idx]:
            loader = 'train_' + loader_type + '_loader'
            if len(self.config['data'][loader_type]) == 1:
                self.config['data'][loader_type].extend([self.config['train']['epochs']])
            for i in range(*self.config['data'][loader_type]):
                self.train_schedule[i] = loader
            if loader in self.train_schedule.values():
                self.data_loaders.update({loader: self.get_dataloaders(train_df, valid_df, loader)})
        # Print schedule
        printlog("Training schedule created:")
        start, stop = None, None
        for i in range(1, self.config['train']['epochs']):
            if start is None:
                start = i - 1
            if self.train_schedule[i] != self.train_schedule[i - 1]:
                stop = i - 1
            elif i == self.config['train']['epochs'] - 1:
                stop = i
            if start is not None and stop is not None:
                if start == stop:
                    printlog("          Epoch {}: {}".format(start, self.train_schedule[i - 1]))
                else:
                    printlog("    Epochs {} to {}: {}".format(start, stop, self.train_schedule[i - 1]))
                start, stop = None, None

    def get_seg_dataframes(self):
        """Creates the training and validation segmentation datasets from the config"""
        if self.dataset in ['CITYSCAPES', 'PASCALC', 'ADE20K', 'IACL']:
            print(f'no dataframes for dataset {self.dataset}')
            return None, None
        elif self.dataset in ['CADIS']:
            # we use csv files for cadis
            train, valid = get_cadis_dataframes(self.config)
            return train, valid

    def get_dataloaders(self, train_df: Union[pd.DataFrame, None], valid_df: Union[pd.DataFrame, None],  mode: str = 'default', **kwargs):
        transforms_dict = dict()
        transforms_dict['train'] = parse_transform_lists(self.config['data']['transforms'],
                                                         self.config['data']['transform_values'],
                                                         self.dataset, self.experiment)
        transforms_dict['valid'] = parse_transform_lists(self.config['data']['transforms_val'],
                                                         self.config['data']['transform_values_val'],
                                                         self.dataset, self.experiment)
        if 'torchvision_normalise' in self.config['data']['transforms']:
            self.un_norm = un_normalise
        else:
            self.un_norm = do_nothing
        # Dataset transforms console output
        img_transforms = [str(type(item).__name__) for item in transforms_dict['train']['img'] if
                          not (isinstance(item, ToPILImage) or isinstance(item, ToTensor))]
        common_transforms = [str(type(item).__name__) for item in transforms_dict['train']['common']]
        printlog("Dataset transforms: {}".format(img_transforms + common_transforms))
        data_path = None if self.data_preloaded else self.config['data_path']
        real_num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]) - 1 \
            if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] \
            else len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1])
        num_workers = int(self.config['data']['num_workers'])

        if self.dataset == 'CITYSCAPES':
            train_split = self.config['data']['split'][0] if not self.debugging else 'val'
            train_set = cts(root=data_path, split=train_split, mode='fine', target_type='semantic',
                            transforms_dict=transforms_dict['train'])
            valid_split = self.config['data']['split'][1] if not self.debugging else 'val'
            valid_set = cts(root=data_path, split=valid_split, mode='fine', target_type='semantic',
                            transforms_dict=transforms_dict['valid'])

            if self.dataset == 'CITYSCAPES' and self.save_outputs and self.config['mode'] in ['inference', 'training']:
                 valid_set.return_filename = True

            printlog(f'Cityscapes dataset train on [{train_split}] valid on [{valid_split}]')

            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, rank = self.rank, num_replicas=self.n_gpus)
            else:
                self.train_sampler = None
            train_loader = DataLoader(train_set, batch_size=self.batch_size, drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            # self.batches_per_epoch = len(train_loader)
            # todo figure out what to do with valid_loader should not be created in every worker
            valid_loader = DataLoader(valid_set, batch_size=self.valid_batch_size,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)
            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                  "".format(len(train_set), len(valid_set)))
            printlog("Dataloaders created. Batch size: {}\n"
                  "              Number of workers: {}\n"
                  .format(self.batch_size, num_workers))
            return train_loader, valid_loader

        if self.dataset == 'PASCALC':
            # todo: only default mode for PASCALC is currently implemented

            train_split = self.config['data']['split'][0]

            train_set = pc(root=data_path, split=train_split, transforms_dict=transforms_dict['train'])

            valid_split = self.config['data']['split'][1]

            valid_set = pc(root=data_path, split=valid_split, transforms_dict=transforms_dict['valid'])

            if self.save_outputs and self.config['mode'] in ['inference', 'training']:
                 valid_set.return_filename = True

            printlog(f'PASCALC dataset train on [{train_split}] valid on [{valid_split}]')

            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, rank = self.rank, num_replicas=self.n_gpus)
            else:
                self.train_sampler = None
            train_loader = DataLoader(train_set, batch_size=self.batch_size, drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            # self.batches_per_epoch = len(train_loader)
            # todo figure out what to do with valid_loader should not be created in every worker
            valid_loader = DataLoader(valid_set, batch_size=self.valid_batch_size,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)
            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                  "".format(len(train_set), len(valid_set)))
            printlog("Dataloaders created. Batch size: {}\n"
                  "              Number of workers: {}\n"
                  .format(self.batch_size, num_workers))
            return train_loader, valid_loader

        if self.dataset == 'ADE20K':
            # todo: only default mode for ADE20K is currently implemented
            train_split = self.config['data']['split'][0]

            train_set = ade(root=data_path, split=train_split, transforms_dict=transforms_dict['train'])

            valid_split = self.config['data']['split'][1]

            valid_set = ade(root=data_path, split=valid_split, transforms_dict=transforms_dict['valid'])

            if self.save_outputs and self.config['mode'] in ['inference', 'training']:
                 valid_set.return_filename = True

            printlog(f'ADE20K dataset train on [{train_split}] valid on [{valid_split}]')

            if self.parallel:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, rank = self.rank, num_replicas=self.n_gpus)
            else:
                self.train_sampler = None
            train_loader = DataLoader(train_set, batch_size=self.batch_size, drop_last=True,
                                      pin_memory=True,
                                      sampler=self.train_sampler,
                                      shuffle=self.train_sampler is None,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)

            # self.batches_per_epoch = len(train_loader)
            # todo figure out what to do with valid_loader should not be created in every worker
            valid_loader = DataLoader(valid_set, batch_size=self.valid_batch_size,
                                      num_workers=num_workers, worker_init_fn=worker_init_fn)
            printlog("Dataset split created. Number of records training / validation: {:06d} / {:06d}\n"
                  "".format(len(train_set), len(valid_set)))
            printlog("Dataloaders created. Batch size: {}\n"
                  "              Number of workers: {}\n"
                  .format(self.batch_size, num_workers))
            return train_loader, valid_loader

        if self.dataset == 'CADIS':
            if mode == 'default':
                train_set = DatasetFromDF(train_df, self.experiment, transforms_dict['train'],
                                          dataset=self.dataset,data_path=data_path)
                valid_set = DatasetFromDF(valid_df, self.experiment, transforms_dict['valid'],
                                          dataset=self.dataset,data_path=data_path)
                if self.parallel:
                    self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, rank=self.rank,
                                                                                         num_replicas=self.n_gpus)
                else:
                    self.train_sampler = None
                train_loader = DataLoader(train_set, batch_size=self.batch_size, sampler=self.train_sampler,
                                          shuffle=self.train_sampler is None, num_workers=num_workers, worker_init_fn=worker_init_fn)
                # self.batches_per_epoch = len(train_loader)
                valid_loader = DataLoader(valid_set, num_workers=num_workers, worker_init_fn=worker_init_fn)
                printlog("Dataloaders created. Batch size: {}\n"
                      "              Number of workers: {}\n"
                      .format(self.batch_size, num_workers))
                return train_loader, valid_loader
            elif mode == 'train_repeat_factor_loader':

                train_set = DatasetFromDF(train_df, self.experiment, transforms_dict['train'],
                                          data_path=data_path)

                self.train_sampler = RepeatFactorSampler(data_source=train_set, dataframe=train_df,
                                                         repeat_thresh=self.config['data']['repeat_factor_freq_thresh'],
                                                         experiment=self.config['data']['experiment'],
                                                         split=int(self.config['data']['split']))

                batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=self.train_sampler,
                                                                      batch_size=self.batch_size,
                                                                      drop_last=True)

                train_repeat_factor_loader = DataLoader(train_set,
                                                        batch_sampler=batch_sampler,
                                                        num_workers=self.config['data']['num_workers'],
                                                        worker_init_fn=worker_init_fn)
                # valid_set = DatasetFromDF(valid_df, self.experiment, transforms_dict['valid'], data_path=data_path)
                # valid_loader = DataLoader(valid_set, num_workers=num_workers, worker_init_fn=worker_init_fn)
                img_rfs = self.train_sampler.repeat_factors.numpy()
                # cls_rfs = sampler.class_repeat_factors
                # cls_rfs = {k: v for k, v in sorted(cls_rfs.items(), reverse=True, key=lambda item: item[1])}
                frames_repeated = sum(img_rfs[img_rfs > 1])
                printlog("Repeat factor dataloader created. frequency threshold: {:.2f}\n"
                      "                                  frames with rf>1:  {}\n"
                      "                      Resulting total training records (aprox): {:.2f}"
                      .format(self.config['data']['repeat_factor_freq_thresh'], frames_repeated, sum(img_rfs)))
                return train_repeat_factor_loader
            else:
                ValueError("Dataloader special type '{}' not recognised".format(mode))

    def load_model(self):
        """Loads the model into self.model"""
        model_class = globals()[self.config['graph']['model']]
        self.model = model_class(config=self.config['graph'], experiment=self.experiment)
        out_stride = self.model.out_stride
        if hasattr(self.model, 'projector_model'):
            self.return_features = self.model.projector_model is not None and self.config['mode'] == 'training'
        else:
            self.return_features = False

        if self.parallel:
            self.model.cuda()
            printlog(f"using sync batch norm : {self.config['graph']['sync_bn']}")
            if self.config['graph']['sync_bn']:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            else:
                printlog(f"using sync batch norm : {False}")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device],
                                                                   gradient_as_bucket_view=True)
        else:
            self.model = self.model.to(self.device)

        num_train_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        printlog("Using model '{}' with backbone '{}' with output stride {} : trainable parameters {}"
              .format(self.config['graph']['model'], self.config['graph']['backbone'],  out_stride, num_train_params))
        if 'graph' in self.config:
            # todo change config of upernet to have all model architecture info under 'graph' -- to avoid ifs
            if 'ss_pretrained' in self.config['graph']:
                if self.config['graph']['ss_pretrained']:
                    self.load_ss_pretrained()

    def load_loss(self):
        """Load loss function"""
        assert 'loss' in self.config
        if 'loss' in self.config:
            self.config['loss']['experiment'] = self.experiment
            self.config['loss']['device'] = str(self.device)
            loss_class = globals()[self.config['loss']['name']]

            if self.config['loss']['name'] == 'CrossEntropyLoss':
                ignore_index_in_loss = len(
                    DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]) - 1 \
                    if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] else -100
                self.loss = CrossEntropyLoss(ignore_index=ignore_index_in_loss)
            else:
                self.loss = loss_class(self.config['loss'])

            self.loss = self.loss.to(self.device)

            if isinstance(self.loss, LossWrapper):
                if any(['DenseContrastive' in term for term in self.loss.loss_classes]):
                    if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                        assert hasattr(self.model.module, 'projector_model') or hasattr(self.model.module, 'projector'), \
                            'model must have projector if DC loss is used'
                        assert self.model.module.projector_model is not None, \
                            'model must have projector if DC loss is used'
                    else:
                        assert hasattr(self.model, 'projector_model') or hasattr(self.model, 'projector'), \
                            'model must have projector if DC loss is used'
                        assert self.model.projector_model is not None,\
                            'model must have projector if DC loss is used'

                printlog(f"Loaded loss: {self.loss.info_string} rank : {self.rank}")
            else:
                printlog(f"Loaded loss function: {loss_class} rank : {self.rank}")

    def load_optimiser(self):
        """Set optimiser and if required, learning rate schedule"""
        params = self.model.parameters()
        if 'optim' not in self.config['train']:
            printlog('defaulting to adam optimiser')
            self.config['train']['optim'] = 'Adam'
            self.optimiser = torch.optim.Adam(params, lr=self.config['train']['learning_rate'])
        else:
            if 'opt_keys' in self.config['train']:
                params = get_param_groups_using_keys(self.model, self.config)
            elif 'stage_wise_lr' in self.config['train']:
                params = get_param_groups_with_stage_wise_lr_decay(self.model, self.config)
            if self.config['train']['optim'] == 'SGD':
                wd = self.config['train']['weight_decay'] if 'weight_decay' in self.config['train'] else 0.0005
                momentum = self.config['train']['momentum'] if 'momentum' in self.config['train'] else 0.9
                self.optimiser = torch.optim.SGD(params, lr=self.config['train']['learning_rate'],
                                                 momentum=momentum, weight_decay=wd)
            elif self.config['train']['optim'] == 'Adam':
                self.optimiser = torch.optim.Adam(params, lr=self.config['train']['learning_rate'])
            elif self.config['train']['optim'] == 'AdamW':
                wd = self.config['train']['weight_decay'] if 'weight_decay' in self.config['train'] else 0.01
                betas = self.config['train']['betas'] if 'momentum' in self.config['train'] else (0.9, 0.999)
                self.optimiser = torch.optim.AdamW(params, lr=self.config['train']['learning_rate'],
                                                   betas=betas, weight_decay=wd)
            else:
                raise ValueError(f"optimizer {self.config['train']['optim']} not recognized")

        if self.config['train']['lr_batchwise']:  # Replace given lr_restarts with numbers in batches instead of epochs
            batches_per_epoch = [len(self.data_loaders[self.train_schedule[e]]) for e in range(self.config['train']['epochs'])]
            lr_total_steps = np.sum(batches_per_epoch)
            r = self.config['train']['lr_restarts']
            new_r = []
            if len(r) > 0:
                r.insert(0, 0)
                for i in range(len(r) - 1):
                    new_r.append(int(np.sum(np.array(batches_per_epoch)[r[i]:r[i + 1]]) + np.sum(new_r[:i])))
            lr_restart_steps = new_r
            # # Adjust params for exponential decay. This is experimental - adjustment is such that the decay over steps
            # # in the very first epoch will equal a decay of lr_params in that first epoch. If later epochs use a
            # # different number of steps, this is not taken into account.
            # self.config['train']['lr_params'] = np.power(self.config['train']['lr_params'], 1 / b_per_e[0])
        else:
            lr_restart_steps = self.config['train']['lr_restarts']
            lr_total_steps = self.config['train']['epochs']

        printlog("*** lr_schedule: '{}' over total steps {} with restarts @ {} batchwise_schedule {}"
              .format(self.config['train']['lr_fct'],
                      lr_total_steps, lr_restart_steps,
                      self.config['train']['lr_batchwise']))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=LRFcts(self.config['train'],
                                                                                            lr_restart_steps,
                                                                                            lr_total_steps))

        # lrs = []
        # for i in range(lr_total_steps):
        #     lrs.append(self.scheduler.get_lr())
        #
        #     if i <= lr_total_steps-2:
        #         self.scheduler.step()
        #
        # import matplotlib.pyplot as plt
        # lr_funct = plt.plot(lrs)
        # plt.show()
        # a=1

    def train_one_epoch(self):
        """Train the model for one epoch"""
        raise NotImplementedError

    def validate(self):
        """Validate the model on the validation data"""
        raise NotImplementedError

    def post_process_output(self, **kwargs):
        """Validate the model on the validation data"""
        raise NotImplementedError

    def forward_step(self, **kwargs):
        " all forward computation using self.model , including using self.loss "
        raise NotImplementedError

    def infer(self):
        assert 'load_checkpoint' in self.config, 'load_checkpoint: "run_id" must be in config for inference mode!'
        """run the model on validation data of a split , creates a logfile named 'infer' in logging dir """

        # prep model flags
        self.model.eval()
        if hasattr(self.model, 'get_intermediate'):
            self.model.get_intermediate = False  # to supress ocr output
        elif hasattr(self.model, 'module'):
            if hasattr(self.model.module, 'get_intermediate'):
                self.model.module.get_intermediate = False

        if hasattr(self.model, 'return_features'):
            self.model.return_features = False # to supress features from projector
        elif hasattr(self.model, 'module'):
            if hasattr(self.model.module, 'return_features'):
                self.model.module.return_features = False

        # checkpoint load
        if 'load_last' in self.config:
            chkpt_type = 'last' if self.config['load_last'] else 'best'
            self.load_checkpoint(chkpt_type)
        else:
            self.load_checkpoint('best')

        # tta
        tta, json_tag = '', ''
        if self.config['tta']:
            tta = '_tta'
            # protocol used in cityscapes results
            scales_list = [0.75, 1.25, 1.5, 1.75, 2] if not self.debugging else [1.0]
            if 'tta_scales' in self.config and not self.debugging:
                scales_list = self.config["tta_scales"]
            json_tag += f'_{scales_list}_flip'

            if self.dataset == 'CITYSCAPES':
                crop_size = self.config['data']['transform_values']['crop_shape']
                strides = self.config['strides'] if 'strides' in self.config else crop_size
                flip = self.config['flip'] if 'flip' in self.config else True
                tta_model = TTAWrapperCTS(self.model, scales_list, flip, strides, crop_size, self.debugging)

            elif self.dataset == 'PASCALC':
                tta_model = TTAWrapperPC(self.model, scales_list)
            elif self.dataset == 'ADE20K' and 'strides' in self.config:
                crop_size = self.config['data']['transform_values']['crop_shape']
                strides = self.config['strides'] if 'strides' in self.config else crop_size
                flip = self.config['flip'] if 'flip' in self.config else True
                tta_model = TTAWrapperSlide(self.model, scales_list, flip, strides, crop_size, self.debugging)
            else:
                tta_model = TTAWrapper(self.model, scales_list)
            printlog(f'** {tta_model.__class__.__name__}using tta with transforms \n ** flip and scales: {tta_model.scales}')

        confusion_matrix = None
        with torch.no_grad():
            for rec_num, (img, lbl, metadata) in enumerate(self.data_loaders['valid_loader']):
                print("\r Inference on {}".format(rec_num), end='', flush=True)
                img, lbl = img.to(self.device), lbl.to(self.device)
                output = self.model(img.float()) if not self.config['tta'] else tta_model(img.float())
                img, output, lbl = self.post_process_output(img=img, output=output, lbl=lbl, metadata=metadata)

                confusion_matrix = t_get_confusion_matrix(output, lbl, self.dataset, confusion_matrix)
                # if rec_num in np.round(np.linspace(0, len(self.data_loaders['valid_loader']) - 1, self.max_valid_imgs)):
                #     if not isinstance(self.model, Ensemble):
                #         lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
                #     else:
                #         lbl_pred = torch.argmax(output, dim=1)  # already softmaxed and merged in ensemble.forward()
                #     self.valid_writer.add_image(
                #         'valid_images/record_{:02d}'.format(rec_num),
                #         to_comb_image(self.un_norm(img)[0], lbl[0], lbl_pred[0], self.config['data']['experiment'], self.dataset),
                #         self.global_step, dataformats='HWC')

                if self.save_outputs:
                    self.save_output(output, metadata)
                    print("\r saved {}".format(rec_num), end='', flush=True)

                if rec_num == 10 and self.debugging:
                    print(f'stopping at {rec_num}')
                    break

        mious = t_get_mean_iou(confusion_matrix, self.config['data']['experiment'], self.dataset, True, rare=True)
        split = self.config['data']['split'][-1] if self.dataset in ['CITYSCAPES', 'PASCALC', 'ADE20K'] else 'val'
        self.write_dict_json(config=mious, filename=f'{self.date}_infer{tta}{json_tag}_split_{split}')
        # logging
        self.valid_writer.add_scalar('metrics/mean_iou', mious['mean_iou'], self.global_step)
        msg_str = "\rmiou:{:.4f} ".format(mious['mean_iou'])
        for categ in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][2]:
            self.valid_writer.add_scalar('metrics/{}'.format(categ), mious['categories'][categ], self.global_step)
            msg_str += "- {}:{:.4f}".format(categ, mious['categories'][categ])
        printlog(msg_str)
        printlog(mious['per_class_iou'])
        self.valid_writer.close()

    def save_output(self, output, metadata):
        filename = metadata['target_filename'][0]
        pred = torch.argmax(nn.Softmax2d()(output), dim=1)  # contains train_ids need class_ids for evaluation
        split = self.config['data']['split'][-1] if self.dataset in ['CITYSCAPES', 'PASCALC', 'ADE20K'] else 'val'
        if self.dataset == 'ADE20K' and split=='test':
            filename =  metadata['img_filename'][0]
        create_new_directory(str(pathlib.Path(self.log_dir) / 'outputs' / split / 'debug'))
        create_new_directory(str(pathlib.Path(self.log_dir) / 'outputs' / split / 'submit'))
        debug_pred = mask_to_colormap(to_numpy(pred)[0],
                                      get_remapped_colormap(
                                          DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0],
                                          self.dataset),
                                      from_network=True, experiment=self.experiment,
                                      dataset=self.dataset)[..., ::-1]
        cv2.imwrite(
            str(pathlib.Path(self.log_dir) / 'outputs' / split / 'debug' / pathlib.Path(filename).stem) + '.png',
            debug_pred)
        class_to_train_mapping = DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0]
        train_to_class_mapping = reverse_mapping(class_to_train_mapping)
        submission_pred = remap_mask(to_numpy(pred)[0], train_to_class_mapping)
        cv2.imwrite(
            str(pathlib.Path(self.log_dir) / 'outputs' / split / 'submit' / pathlib.Path(filename).stem) + '.png',
            submission_pred)

    def demo_tsne(self):
        assert 'load_checkpoint' in self.config, 'load_checkpoint: "run_id" must be in config for demo_tsne mode!'
        """run the model on validation data of a split , creates a logfile named 'infer' in logging dir """
        from utils import TsneMAnager
        # prep model flags
        self.model.eval()
        # if hasattr(self.model, 'get_intermediate'):
        #     self.model.get_intermediate = False  # to supress ocr output
        # elif hasattr(self.model, 'module'):
        #     if hasattr(self.model.module, 'get_intermediate'):
        #         self.model.module.get_intermediate = False

        if hasattr(self.model, 'return_features'):
            self.model.return_features = True  # to supress features from projector
        elif hasattr(self.model, 'module'):
            if hasattr(self.model.module, 'return_features'):
                self.model.module.return_features = True
        # checkpoint load
        if 'load_last' in self.config:
            chkpt_type = 'last' if self.config['load_last'] else 'best'
            self.load_checkpoint(chkpt_type)
        else:
            self.load_checkpoint('best')
        self.TsneManager = TsneMAnager(self.dataset,
                                       self.model.num_classes,
                                       feat_dim=48,
                                       scale=self.config['tsne_scale'],
                                       run_id=self.run_id)
        s = 0
        # print('computing_tsne for feats of dim'
        if self.config['graph']['model']=='UPerNet' and self.config['graph']['backbone']=='resnet101':
            if self.TsneManager.scale == 4:
                s = 0
                self.TsneManager.feat_dim = 256
            elif self.TsneManager.scale == 8:
                s = 1
                self.TsneManager.feat_dim = 512
            elif self.TsneManager.scale == 16:
                s = 2
                self.TsneManager.feat_dim = 1024
            elif self.TsneManager.scale == 32:
                s = 3
                self.TsneManager.feat_dim = 2048
        elif self.config['graph']['model']=='HRNet':
            if self.TsneManager.scale == 4:
                s = 0
                self.TsneManager.feat_dim = 48
            elif self.TsneManager.scale == 8:
                s = 1
                self.TsneManager.feat_dim = 96
            elif self.TsneManager.scale == 16:
                s = 2
                self.TsneManager.feat_dim = 192
            elif self.TsneManager.scale == 32:
                s = 3
                self.TsneManager.feat_dim = 384
        else:
            raise NotImplementedError()
        print(f'computing tsne for feats of scale : {self.TsneManager.scale} dim : {self.TsneManager.feat_dim}')
        # tta
        tta, json_tag = '', ''
        confusion_matrix = None
        feats_for_tsne = []
        with torch.no_grad():
            for rec_num, (img, lbl, metadata) in enumerate(self.data_loaders['valid_loader']):
                print("\r Processing {}".format(rec_num), end='', flush=True)
                img, lbl = img.to(self.device), lbl.to(self.device)
                output, features = self.model(img.float())
                if self.config['graph']['model']=='HRNet':
                    f = features[-1][s]
                elif self.config['graph']['model'] == 'UPerNet':
                    f = features[::-1][s]
                self.TsneManager.accumulate(f, lbl)

                # feats_for_tsne.append(features[0]) # take only s4 feats from hrnet
                # img, output, lbl = self.post_process_output(img=img, output=output, lbl=lbl, metadata=metadata)
                if rec_num == 100 and self.debugging:
                    print(f'stopping at {rec_num}')
                    feats = self.TsneManager.compute(self.log_dir)
                    break
            if not self.debugging:
                _ = self.TsneManager.compute(self.log_dir)
            # features_tsne = get_tsne_embedddings_ms(feats_for_tsne, lbl, scale=4, dataset=self.dataset)
