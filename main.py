import argparse
# noinspection PyUnresolvedReferences
from managers import *
from utils import parse_config


def str2bool(s:str):
    assert type(s), f'input argument must be str instead {s}'
    if s in ['True', 'true']:
        return True
    elif s in ['False', 'false']:
        return False
    else:
        raise ValueError(f'string {s} ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, default='configs/FCN_train_config.json',
                        help='Set path to configuration files, e.g. '
                             'python main.py --config configs/FCN_train_config.json.')

    parser.add_argument('-u', '--user', type=str, default='c',
                        help='Select user to set correct data / logging paths for your system, e.g. '
                             'python main.py --user theo')

    parser.add_argument('-d', '--device',  nargs="+", type=int, default=-1,
                        help='Select GPU device to run the experiment one.g. --device 3')

    parser.add_argument('-s', '--dataset', type=str, default=-1, required=False,
                        help='Select dataset to run the experiment one.g. --device 3')

    parser.add_argument('-p', '--parallel', action='store_true',
                        help='whether to use distributed training')

    parser.add_argument('-debug', '--debugging', action='store_true',
                        help='sets manager into debugging mode e.x --> cts is run with val/val split')

    parser.add_argument('-cdnb', '--cudnn_benchmark', type=str, default=None, required=False,
                        help='if added in args then uses cudnn benchmark set to True '
                             'else uses config '
                             'else sets it to True by default')

    parser.add_argument('-cdne', '--cudnn_enabled', type=str, default=None, required=False,
                        help='if added in args then uses cudnn enabled set to True '
                             'else uses config '
                             'else sets it to True by default')

    parser.add_argument('-vf', '--valid_freq', type=int, default=None, required=False,
                        help='sets how often to run validation')

    parser.add_argument('-w', '--workers', type=int, default=None, required=False,
                        help='workers for dataloader per gpu process')

    parser.add_argument('-ec', '--empty_cache', action='store_true',
                        help='whether to empty cache (per gpu process) after each forward step to avoid OOM --'
                             ' this is useful in DCV2_ms or DCV3/ms')

    parser.add_argument('-m', '--mode', type=str, default=None, required=False,
                        help='mode setting e.x training, inference (see BaseManager for others)')

    parser.add_argument('-cpt', '--checkpoint', type=str, default=None, required=False,
                        help='path to checkpoint folder')

    parser.add_argument('-bs', '--batch_size', type=int, default=None, required=False,
                        help='batch size -- the number given is then divided by n_gpus if ddp')

    parser.add_argument('-ep', '--epochs', type=int, default=None, required=False,
                        help='training epochs -- overrides config')

    parser.add_argument('-so', '--save_outputs', action='store_true',
                        help='whether to save outputs for submission cts')

    parser.add_argument('-rfv', '--run_final_val', action='store_true',
                        help='whether to run validation with special settings'
                             ' at the end of training (ex using tta or sliding window inference)')

    parser.add_argument('-tta', '--tta', action='store_true',
                        help='whether to tta_val at the end of training')

    parser.add_argument('-tsnes', '--tsne_scale', type=int, default=None, required=False,
                        help=' stride of feats on which to apply tsne must be [4,8,16,32]')

    # loss args for convenience
    parser.add_argument('--loss', '-l', choices=[None,'ce', 'ms', 'ms_cs'], default=None, required=False,
                        help=f'choose loss overriding config (refer to config for other options except {"[ce, ms, ms_cs]"}')

    args = parser.parse_args()
    config = parse_config(args.config, args.user, args.device, args.dataset, args.parallel)
    manager_class = globals()[config['manager'] + 'Manager']
    print(f'requested device ids:  {config["gpu_device"]}')
    print('parsing cmdline args')
    # override config
    config['parallel'] = args.parallel
    config['tsne_scale'] = args.tsne_scale
    if args.loss:
        print(f'overriding loss type in config requested [{args.loss}]')
        if 'ms' in args.loss:
            config['loss'].update({"losses": {"CrossEntropyLoss": 1, "DenseContrastiveLossV2_ms": 0.1}})
            config['loss'].update({"cross_scale_contrast": False})
            if config['graph']['model'] == 'UPerNet':
                config['graph'].update({"ms_projector": {"mlp": [[1, -1, 1]],  "scales":4, "d": 256, "use_bn": True, "position":"backbone"}})
            else:
                config['graph'].update({"ms_projector": {"mlp": [[1, -1, 1]],  "scales":4, "d": 256, "use_bn": True}})

        if 'cs' in args.loss:
            config['loss'].update({"cross_scale_contrast": True})

        if args.loss == 'ce':
            config['loss'].update({"losses": {"CrossEntropyLoss": 1}})
            if 'ms_projector' in config['graph']:
                del config['graph']['ms_projector']

    if args.save_outputs:
        config['save_outputs'] = True
    if args.run_final_val:
        config['run_final_val'] = True
        print('going to run tta val at the end of training')
    if args.empty_cache:
        config['empty_cache'] = True
        print('emptying cache')
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
        print(f'bsize {args.batch_size}')
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs
        print(f'epochs : {args.epochs}')
    if args.tta:
        config['tta'] = True
        print(f'tta set to {config["tta"]}')
    if args.debugging:
        config['debugging'] = True
    if args.valid_freq is not None:
        config['valid_freq'] = args.valid_freq
    if args.workers is not None:
        config['data']['num_workers'] = args.workers
        print(f'workers {args.workers}')
    if args.mode is not None:
        config['mode'] = args.mode
        print(f'mode {args.mode}')
    if args.checkpoint is not None:
        config['load_checkpoint'] = args.checkpoint
        print(f'load_checkpoint set to {args.mode}')

    if args.cudnn_benchmark is not None:
        config['cudnn_benchmark'] = str2bool(args.cudnn_benchmark)
    if args.cudnn_enabled is not None:
        config['cudnn_enabled'] = str2bool(args.cudnn_enabled)

    manager = manager_class(config)

    if config['mode'] == 'training' and not manager.parallel:
        manager.train()
    elif config['mode'] == 'inference':
        manager.infer()
    elif config['mode'] == 'demo_tsne':
        manager.demo_tsne()
    elif config['mode'] == 'submission_inference':
        manager.submission_infer()
