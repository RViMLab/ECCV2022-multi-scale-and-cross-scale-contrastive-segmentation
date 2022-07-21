import pathlib
import json
import torch
import datetime
import numpy as np
from torch import nn
from utils import DATASETS_INFO,  get_matrix_fig, get_log_name, to_comb_image, to_numpy, t_normalise_confusion_matrix,\
    get_rank, printlog, shorten_key
import os
from utils import Logger as Log


class LoggingManager():
    def __init__(self, configuration):
        """
        Sets up the run manager, either for training or for inference
        :param configuration: dict with the following keys:
            mode: 'training' or 'inference' or 'video_inference'
            graph: dict with key 'model' (optionally 'width' etc.), 'loss' (optionally 'multi_scale_loss' etc.)
            data: dict with keys 'named_dataset'/'dataset_list', 'train_split', 'batch_size'
            train: dict with keys 'learning_rate', 'epochs'
            data_path: Base path to where the data is found (in original CaDISv2 format with folders etc.)
            log_path: Base path to where checkpoints etc are logged
            log_every_n_epochs: How often a checkpoint is logged
            cuda: 'true'/'false' - whether the model runs on GPU or CPU
            gpu_device: if cuda==True, then which device is to be used
            seed: torch random seed
            infer_config keys: mode: 'inference', model, width, log_path, log_name, checkpoint_type,
                               cuda, gpu_device (if cuda)
        """
        # Set up parameters
        self.rank = 0
        self.config = configuration
        self.parallel = self.config['parallel']
        self.debugging = self.config['debugging']
        self.start_epoch = 0
        self.epoch = 0
        self.best_loss = 1e10
        self.global_step = 0
        self.max_valid_imgs = self.config['max_valid_imgs']  # Maximum number of images saved to tensorboard
        self.valid_freq = self.config['valid_freq'] if 'valid_freq' in self.config else 1
        self.dataset = self.config['data']['dataset']
        self.config['graph'].update({'dataset': self.dataset})
        self.experiment = self.config['data']['experiment']
        self.metrics = {'best_miou': 0,
                        'best_miou_epoch_step': [0, 0],
                        'best_loss_miou': 0,
                        'best_loss_epoch_step': [0, 0],
                        'final_miou':0,
                        'final_miou_step':0}
        for categ in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][2]:
            self.metrics.update({'best_miou_{}'.format(categ): 0})
            self.metrics.update({'best_loss_miou_{}'.format(categ): 0})
        self.data_preloaded = self.config['data']['preload']
        self.num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0])
        self.model = None
        self.tta_model = None
        self.data_loaders = {}
        self.loss = 0
        self.batch_size = self.config['data']['batch_size']
        self.batches_per_epoch = 100
        self.valid_batch_size = self.config['valid_batch_size'] if 'valid_batch_size' in self.config else 1
        if 'loss' in self.config:
            self.config['loss'].update({'dataset': self.dataset})
            self.config['loss'].update({'experiment': self.experiment})
        self.optimiser = None
        self.scheduler = None
        self.train_schedule = {}
        self.save_dir_path = None  # path to where pseudo labelled data are saved
        self.save_outputs = False

        for i in range(self.config['train']['epochs']):
            self.train_schedule.update({i: 'train_loader'})  # pre-fill

        # Print debugging state in Console
        if self.debugging:
            print("\n\n* * * * * DEBUGGING ACTIVE * * * * * \n\n")
            print(f"** changing num_workers to 0 from {self.config['data']['num_workers']}")
            self.config['data']['num_workers'] = 0

        # Identify run
        self.date = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
        if 'load_checkpoint' in self.config and self.config['mode'] is not 'training':
            self.run_id = self.config['load_checkpoint']
        else:
            self.run_id = '{:%Y%m%d_%H%M%S}_e{}'.format(datetime.datetime.now(), self.experiment)
            if 'name' in self.config:
                self.run_id = '__'.join((self.run_id, get_log_name(self.config)))
        self.log_dir = pathlib.Path(self.config['log_path']) / self.run_id
        if not self.log_dir.is_dir():
            self.log_dir.mkdir(parents=True)

        self.log_file = str(self.log_dir / pathlib.Path(f"{self.run_id}_{self.config['mode']}"))
        printlog(f'loggging in {self.log_file}')
        Log.init(logfile_level="info",
                 stdout_level=None,
                 log_file=self.log_file,
                 rewrite=True)
        self.run_final_val = self.config['run_final_val'] if 'run_final_val' in self.config else False
        printlog(f'going to run tta val after training {self.run_final_val}')
        printlog("Run ID: {} {}".format(self.run_id, self.config['mode']))
        if self.config['mode'] in ['inference', 'training']:
            self.save_outputs = self.config['save_outputs'] if 'save_outputs' in self.config else False
            printlog(f'going to save inference outputs ') if self.save_outputs else None

        # Set cuda flag
        if torch.cuda.is_available() and not self.config['cuda']:
             printlog("CUDA device available, but not used")
        if self.config['cuda'] and not torch.cuda.is_available():
            printlog("CUDA device required, but not available - using CPU instead")
        self.cuda = torch.cuda.is_available() & self.config['cuda']

        # cuda device ids identification
        local_devices_count = torch.cuda.device_count()
        printlog(f'available_devices {local_devices_count}')
        if not self.parallel:
            assert len(self.config['gpu_device']) == 1
            self.config['gpu_device'] = self.config['gpu_device'][0]
        elif len(self.config['gpu_device']) == 1:
            printlog(f'ddp requested but only 1 gpu device requested {self.config["gpu_device"]}')
            local_devices_count = torch.cuda.device_count()
            printlog(f'setting it to all available cuda devices {local_devices_count}')
            self.config['gpu_device'] = [i for i in range(local_devices_count)]

        if self.cuda:
            if self.parallel:
                assert(local_devices_count > 1), 'parallel was set to True but devices are {}<2'.format(local_devices_count)
                printlog(f'available_devices {local_devices_count}')
                self.device = torch.device('cuda')
                self.rank = 0  # init
                self.n_gpus = len(self.config['gpu_device']) if isinstance(self.config['gpu_device'], list) else local_devices_count
                self.allocated_devices = [0, 1, 2, 3, 4, 5, 6, 7] if isinstance(self.config['gpu_device'], list) else self.config['gpu_device']
                self.world_size = self.n_gpus  # one machine only
                printlog("Program will run on *****Multi-GPU-CUDA, devices {}*****".format(self.n_gpus))
                """ Initialize the distributed environment. """
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '29500'

            else:
                self.device = torch.device('cuda')
                torch.cuda.set_device(self.config['gpu_device'])
                printlog("Program will run on *****GPU-CUDA, device {}*****".format(self.config['gpu_device']))
        else:
            self.device = torch.device('cpu')
            printlog("Program will run on *****CPU*****")

        self.empty_cache = self.config['empty_cache'] if 'empty_cache' in self.config else False
        printlog(f'Empty cache after model.forward(): {self.empty_cache}')

        self.valid_writer = None # to supress warning
        self.train_writer = None

    def train_logging(self, batch_num, output, img, lbl, mious, loss, pa, pac, b):
        if 'DenseContrastiveLossV2_ms' in self.loss.loss_classes:
            is_cs = self.loss.loss_classes['DenseContrastiveLossV2_ms'].cross_scale_contrast
        else:
            is_cs = False

        # logging
        if self.scheduler is not None and self.config['train']['lr_batchwise']:
            self.train_writer.add_scalar('parameters/learning_rate', self.scheduler.get_lr()[0], self.global_step) \
                if self.rank == 0 else None
        if batch_num == 0 and self.rank == 0:
            rec_num = 0  # Just take the first of the batch (will be random)
            lbl_pred = torch.argmax(nn.Softmax2d()(output), dim=1)
            self.train_writer.add_image(
                'train_images/record_{:02d}'.format(rec_num),
                to_comb_image(self.un_norm(img)[rec_num], lbl[rec_num], lbl_pred[rec_num], self.config['data']['experiment'],
                              self.dataset),
                self.global_step, dataformats='HWC')

        info_string = ''

        if 'train_adaptive_batching_loader' in self.train_schedule.values():
            iou_values = (1 - self.config['data']['adaptive_iou_update']) * self.metrics['iou_values'] + \
                         self.config['data']['adaptive_iou_update'] * to_numpy(mious['mean_iou'])
            self.metrics['iou_values'][:] = iou_values

        if hasattr(self.loss, 'loss_vals'):
            for key in self.loss.loss_vals:
                info_string += ' {} {:.5f}; '.format(str(shorten_key(key, is_cs=is_cs)), self.loss.loss_vals[key].item())

                self.train_writer.add_scalar('metrics/{}'.format(str(key)), self.loss.loss_vals[key].item(),
                                             self.global_step) if self.rank == 0 else None

        if self.rank == 0:
            self.train_writer.add_scalar('metrics/loss', loss.item(), self.global_step)
            self.train_writer.add_scalar('metrics/pixel_accuracy', pa, self.global_step)
            self.train_writer.add_scalar('metrics/pixel_accuracy_per_class', pac, self.global_step)

        if 'DenseContrastiveLossV2' in self.loss.loss_classes:
            if self.loss.loss_classes['DenseContrastiveLossV2'].log_this_step:
                Log.info("Epoch {:03d} iter {:06d}, Batch {:03d} - Loss: {:.5f}; {} t: {:.1f} "
                          "r {} ".format(self.epoch + self.start_epoch, self.global_step, batch_num,
                                                    loss.item(), info_string, b, self.rank))
                self.loss.loss_classes['DenseContrastiveLossV2'].log_this_step = False

        elif 'DenseContrastiveCenters' in self.loss.loss_classes:
            if self.loss.loss_classes['DenseContrastiveCenters'].log_this_step:
                Log.info("Epoch {:03d} iter {:06d}, Batch {:03d} - Loss: {:.5f}; {} t: {:.1f} "
                         "r {} ".format(self.epoch + self.start_epoch, self.global_step, batch_num,
                                        loss.item(), info_string, b, self.rank))
                self.loss.loss_classes['DenseContrastiveCenters'].log_this_step = False

        # if is_distributed():
        #     loss = reduce_tensor(self.loss)
        print("\rEpoch {:03d} iter {:06d}, Batch {:03d} - Loss: {:.4f}; {} t: {:.1f} r {} ".format(
            self.epoch + self.start_epoch, self.global_step, batch_num, loss.item(), info_string, b, self.rank), end='',
              flush=True)

        self.global_step += 1

    def valid_logging(self, valid_loss, confusion_matrix, individual_losses, mious, pa, pac):
        """ logging - checkpoint saving - best val tracking """
        self.valid_writer.add_scalar('metrics/loss', valid_loss, self.global_step - 1)
        info_string = ''
        if hasattr(self.loss, 'loss_vals'):
            for key in self.loss.loss_vals:
                individual_losses[key] /= len(self.data_loaders['valid_loader'])
                info_string += ' {} {:.5f}; '.format(str(key), individual_losses[key].item())
                self.valid_writer.add_scalar('metrics/{}'.format(str(key)), individual_losses[key].item(),
                                             self.global_step - 1)

        row_confusion_matrix = t_normalise_confusion_matrix(confusion_matrix, 'row')
        col_confusion_matrix = t_normalise_confusion_matrix(confusion_matrix, 'col')
        self.valid_writer.add_figure('valid_confusion_matrix/row_normalised',
                                     get_matrix_fig(to_numpy(row_confusion_matrix), self.config['data']['experiment'], self.dataset),
                                     self.global_step - 1)
        self.valid_writer.add_figure('valid_confusion_matrix/col_normalised',
                                     get_matrix_fig(to_numpy(col_confusion_matrix), self.config['data']['experiment'], self.dataset),
                                     self.global_step - 1)

        self.valid_writer.add_scalar('metrics/pixel_accuracy', pa, self.global_step)
        self.valid_writer.add_scalar('metrics/pixel_accuracy_per_class', pac, self.global_step)
        self.valid_writer.add_scalar('metrics/mean_iou', mious['mean_iou'], self.global_step)

        msg_str = "\rEpoch {:03d} - val loss: {:.5f} - miou:{:.2f} ".format(self.epoch + self.start_epoch, valid_loss, mious['mean_iou'])
        categ_mious = []
        mious_values = dict()
        for categ in mious['categories']:
            categ_mious.append(mious['categories'][categ])
            self.valid_writer.add_scalar('metrics/{}'.format(categ), categ_mious[-1], self.global_step)
            msg_str += "- {}:{:.2f}".format(categ, categ_mious[-1])
            mious_values[categ] = round(float(categ_mious[-1].cpu().numpy()), 4)
        printlog(msg_str)
        m_iou = round(float(mious['mean_iou'].cpu().numpy()), 4)

        best_miou_flag = False
        if m_iou > self.metrics['best_miou']:
            best_miou_flag = True
            self.metrics.update({'best_miou': m_iou,
                                 'best_miou_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})
            msg_str = "            New best mIoU (tot "
            msg1, msg2 = "", "{:.4f} ".format(m_iou)
            for categ in mious['categories']:
                self.metrics.update({'best_miou_{}'.format(categ): mious_values[categ]})
                msg1 += "/ {} ".format(categ)
                msg2 += "{:.4f} ".format(mious_values[categ])
            printlog(msg_str + msg1 + ' ): ' + msg2)

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.metrics.update({'best_loss_miou': m_iou,
                                 'best_loss_epoch_step': [self.epoch + self.start_epoch, self.global_step - 1]})
            printlog("            New best validation loss: {:.5f}".format(valid_loss))
            msg_stra = "         --- with mIoU (tot "
            msg_strb = "         --- best mIoU (tot "
            msg1, msg2 = "", "{:.4f} ".format(m_iou)
            msg3, msg4 = "", "{:.4f} ".format(self.metrics['best_miou'])
            for categ in mious['categories']:
                msg1 += "/ {} ".format(categ)
                msg2 += "{:.4f} ".format(mious_values[categ])
                msg3 += "/ {} ".format(categ)
                self.metrics.update({'best_loss_miou_{}'.format(categ): mious_values[categ]})
                msg4 += "{:.4f} ".format(self.metrics['best_miou_{}'.format(categ)])
            if not best_miou_flag:
                printlog(msg_stra + msg1 + ' ): ' + msg2)
                printlog(msg_strb + msg3 + ' ): ' + msg4)

        if best_miou_flag:
            self.save_checkpoint(is_best=True)
        if self.epoch % self.config['log_every_n_epochs'] == 0 and self.epoch > 0\
                or self.epoch == self.config['train']['epochs'] - 1:
            self.save_checkpoint(is_best=False)

        # Update info.json file so it exists in case the run stops / crashes before self.finalise()
        for categ in mious['categories']:
            self.metrics.update({f'final_miou_{categ}': mious_values[categ]})
        self.metrics['final_miou'] = m_iou
        self.metrics['final_miou_epoch_step'] = [self.epoch + self.start_epoch, self.global_step - 1]
        self.write_info_json()

    def save_checkpoint(self, is_best):
        """Saves a checkpoint in given self.log_dir

        :param is_best: Determines whether the checkpoint is a current best
        """
        base_path = self.log_dir / 'chkpts'
        if not base_path.is_dir():
            base_path.mkdir()
        state = {
            'global_step': self.global_step-1, # -1 because train_logging increments by +1 and followed by valid_logging
            'epoch': self.start_epoch + self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'best_loss': self.best_loss,
            'best_miou': self.metrics['best_miou'],
            'final_miou':self.metrics['final_miou'],
            'final_miou_step': self.metrics['final_miou_step'],
            'is_best': is_best
        }
        if self.scheduler is not None:
            state.update({'scheduler_state_dict': self.scheduler.state_dict()})
        if is_best:
            name = 'chkpt_best.pt'
        else:
            name = 'chkpt_epoch_{:03d}.pt'.format(state['epoch'])
        torch.save(state, base_path / name)
        printlog("Checkpoint saved: {}".format(name))

    def load_checkpoint(self, chkpt_type):
        """Load a model and model state from a checkpoint

        :param chkpt_type: 'best' or 'last'
        :return:
        """
        checkpoint_list = [f.name for f in (self.log_dir / 'chkpts').iterdir()]
        checkpoint_list.sort()
        name = 'chkpt_best.pt'
        if chkpt_type == 'best':
            n = str(self.log_dir / 'chkpts')
            if n not in checkpoint_list:
                printlog("No checkpoint of type 'best' found.")
            elif 'chkpt_epoch_' in checkpoint_list[-1]:
                name = checkpoint_list[-1]
            else:
                raise ValueError(f'Neither chkpt of type "best" nor of type "last" was found'
                                 f' in chekpoints_list {checkpoint_list}')
        elif chkpt_type == 'last':
            if 'chkpt_epoch_' in checkpoint_list[-1]:
                name = checkpoint_list[-1]
            else:
                raise ValueError("No checkpoint of type 'last' found.")
        path = self.log_dir / 'chkpts' / name
        # print(torch.cuda.current_device())
        # this is required if it checkpoint trained on one device and now is loaded on a different device
        # https://github.com/pytorch/pytorch/issues/15541
        if self.parallel:
           map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
        else:
           map_location = 'cuda:{}'.format(self.config['gpu_device'])
        checkpoint = torch.load(str(path), map_location)
        if not self.parallel:
           checkpoint['model_state_dict'] = self.check_module_prefix(checkpoint['model_state_dict'])
        ret = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        printlog(f'loading_state_dict \n :{ret}')
        if self.config['mode'] == 'training':
            printlog(f'loading optimizer_state_dict')
            self.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            printlog(f'loading scheduler_state_dict')
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['global_step']
            self.best_loss = checkpoint['best_loss']
            self.metrics['best_miou'] = checkpoint['best_miou']
            self.metrics['final_miou'] = checkpoint['final_miou'] if 'final_miou' in checkpoint else None
            printlog(f'start_epoch :{self.start_epoch} global_step: {self.global_step}')
        printlog(f"rank : {get_rank()} Checkpoint loaded: {path} type: {chkpt_type}")

    def _check_model_param_prefix(self,state_dict, prefix):
        found_prefix_model = False
        for param_name in state_dict:
            if not param_name.startswith(prefix):
                found_prefix_model = False
                if found_prefix_model:
                    raise Warning('module prefix found in some of the model params but not others '
                                  '-- this will cause bugs!! -- check before proceeding')
                break
            else:
                found_prefix_model = True
        return found_prefix_model

    def check_module_prefix(self, chkpt_state_dict):
        found_prefix_model = self._check_model_param_prefix(self.model.state_dict(), prefix='module.')
        found_prefix_chkpt = self._check_model_param_prefix(chkpt_state_dict, prefix='module.')

        # remove prefix from chkpt_state_dict keys
        if ~found_prefix_model and found_prefix_chkpt:
            for k in list(chkpt_state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    chkpt_state_dict[k[len("module."):]] = chkpt_state_dict[k]
                # delete renamed or unused k
                del chkpt_state_dict[k]

        return chkpt_state_dict

    def finalise(self):
        """Saves info, resets main variables"""
        config_text = self.write_info_json()
        # Save extra info to tensorboard
        self.train_writer.add_text('info', config_text.replace('\n', '  \n'), self.global_step)
        self.train_writer.close()
        self.valid_writer.close()
        # Reset main variables
        # self.run_id = None
        # self.start_epoch = 0
        # self.epoch = 0
        # self.best_loss = 1e10
        # self.metrics = {'best_miou': 0}
        # self.global_step = 0

    def write_info_json(self):
        config = self.config.copy()
        config['run_id'] = self.run_id
        config['best_loss'] = self.best_loss

        metrics = self.metrics.copy()
        for k in metrics.keys():
            if isinstance(self.metrics[k], np.ndarray) or isinstance(metrics[k], torch.Tensor):
                # noinspection PyUnresolvedReferences
                metrics[k] = metrics[k].tolist()
        config['metrics'] = metrics

        if 'queue_fillings' in config:
            config['queue_fillings'] = config['queue_fillings'].tolist()

        # Save config to json
        config_text = json.dumps(config, indent=4, sort_keys=True, default=self.default)
        with open(self.log_dir / 'info.json', 'w') as json_file:
            json_file.write(config_text)
        return config_text

    def write_dict_json(self, config:dict, filename='inference'):
        """write a json to log dir"""
        config['run_id'] = self.run_id
        for k in config:
            if isinstance(config[k], np.ndarray) or isinstance(config[k], torch.Tensor):
                # noinspection PyUnresolvedReferences
                config[k] = config[k].tolist()
        # Save config to json
        config_text = json.dumps(config, indent=4, sort_keys=True, default=self.default)
        with open(self.log_dir / f'{filename}.json', 'w') as json_file:
            json_file.write(config_text)
        return config_text

    @staticmethod
    def default(obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, pathlib.WindowsPath) or isinstance(obj, pathlib.PosixPath):
            return str(obj)
        raise TypeError(f'Not serializable {obj}')

    def run_diagnostic(self):

        assert 'diagnostic_type' in self.config, 'config must have a key: "diagnostic_type" ' \
                                                 'specifying the name of the diangostic'
        diag_name = self.config['diagnostic_type']
        print("Starting diagnostic_type : {}".format(diag_name))

        if diag_name == 'dominant_class':
            def comb_masks(tensor_list, experiment, dataset):
                with torch.no_grad():
                    np_list = []
                    for t in tensor_list:
                        np_a = to_numpy(t)
                        np_a = mask_to_colormap(np_a,
                                                get_remapped_colormap(DATASETS_INFO[dataset].CLASS_INFO[experiment][0], dataset),
                                                from_network=True, experiment=experiment, dataset=dataset)
                        np_list.append(np_a)
                    comb_img = np.concatenate(tuple(np_list), axis=1)
                    return comb_img

            self.config.update({"loss": {
                                    "name": "DenseContrastiveLoss",
                                    "temperature": 0.1,
                                    "dominant_mode": "rare",
                                    "dc_weightings": {
                                         "outer_freq": False,
                                         "outer_entropy": True,
                                         "outer_confusionmatrix": False,
                                         "inner_crossentropy": False,
                                         "inner_idealcrossentropy": False,
                                         "neg_confusionmatrix": False,
                                         "neg_negativity": False
                                    }}})
            self.load_loss()
            scale = 8*2
            dom_mode = self.config['loss']['dominant_mode']

            assert(isinstance(self.loss, DenseContrastiveLoss))
            for batch_num, (img, lbl, metadata) in enumerate(self.data_loaders[self.train_schedule[self.epoch]]):
                _, lbl = img.to(self.device), lbl.to(self.device)
                # b_, h_, w_ = lbl.shape
                class_distr, _, _ = self.loss.get_dist_and_classes(lbl, scale=scale)
                dom_classes = self.loss.get_dominant_classes(class_distr)
                dom_classes_up = torch.repeat_interleave(dom_classes, repeats=scale, dim=2)
                dom_classes_up = torch.repeat_interleave(dom_classes_up, repeats=scale, dim=3)
                # dom_classes_up = dom_classes_up.transpose(dim0=2, dim1=3)
                b, c, h, w = dom_classes.shape
                lbl_down = f.interpolate(lbl.unsqueeze(1).float(), (h, w), mode='nearest')
                rec_num = 0
                self.train_writer.add_image('dominant_vs_label_{}__scale_{}/record_{:02d}'
                                            .format(dom_mode, scale, rec_num),
                                            comb_masks([dom_classes_up[rec_num, rec_num], lbl[rec_num]],
                                                       self.config['data']['experiment'], self.dataset),
                                            batch_num, dataformats='HWC')

                self.train_writer.add_image('dominant_vs_nearest__{}__scale_{}/record_{:02d}'
                                            .format(dom_mode, scale, rec_num),
                                            comb_masks([dom_classes[rec_num, rec_num], lbl_down[rec_num, rec_num]],
                                                       self.config['data']['experiment'], self.dataset),
                                            batch_num, dataformats='HWC')

        elif diag_name == 'augment':
            for batch_num, (img, lbl, metadata) in enumerate(self.data_loaders[self.train_schedule[self.epoch]]):
                # _, lbl = img.to(self.device), lbl.to(self.device)
                rec_num = 0
                raise NotImplementedError()
        else:
            raise NotImplementedError()
