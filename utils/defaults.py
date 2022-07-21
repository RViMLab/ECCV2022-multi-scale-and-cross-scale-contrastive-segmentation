from .datasets_info import CITYSCAPES_INFO, CADIS_INFO, PASCALC_INFO, ADE20K_INFO
import numpy as np
from typing import Any


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


DATASETS_INFO = EasyDict(CADIS=CADIS_INFO, CITYSCAPES=CITYSCAPES_INFO, PASCALC=PASCALC_INFO, ADE20K=ADE20K_INFO)


def get_cityscapes_colormap():
    """
    Returns cityscapes colormap as in paper
    :return: ndarray of rgb colors
    """
    return np.asarray(
        [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81), (128, 64, 128),
         (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30),
         (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
         (0, 0, 70), (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)]
    )


def get_cadis_colormap():
    """
    Returns cadis colormap as in paper
    :return: ndarray of rgb colors
    """
    return np.asarray(
        [
            [0, 137, 255],
            [255, 165, 0],
            [255, 156, 201],
            [99, 0, 255],
            [255, 0, 0],
            [255, 0, 165],
            [255, 255, 255],
            [141, 141, 141],
            [255, 218, 0],
            [173, 156, 255],
            [73, 73, 73],
            [250, 213, 255],
            [255, 156, 156],
            [99, 255, 0],
            [157, 225, 255],
            [255, 89, 124],
            [173, 255, 156],
            [255, 60, 0],
            [40, 0, 255],
            [170, 124, 0],
            [188, 255, 0],
            [0, 207, 255],
            [0, 255, 207],
            [188, 0, 255],
            [243, 0, 255],
            [0, 203, 108],
            [252, 255, 0],
            [93, 182, 177],
            [0, 81, 203],
            [211, 183, 120],
            [231, 203, 0],
            [0, 124, 255],
            [10, 91, 44],
            [2, 0, 60],
            [0, 144, 2],
            [133, 59, 59],
        ]
    )


def get_pascalc_colormap():
    cmap = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
           [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
           [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
           [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
           [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
           [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
           [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
           [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
           [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
           [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
           [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
           [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
           [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
           [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
           [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]
    return cmap


def get_ade20k_colormap():
    # 151 VALUES , CMAP[0] IS IGNORED
    cmap = [[0,0,0], [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
               [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
               [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
               [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
               [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
               [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
               [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
               [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
               [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
               [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
               [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
               [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
               [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
               [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
               [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
               [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
               [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
               [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
               [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
               [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
               [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
               [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
               [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
               [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
               [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
               [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
               [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
               [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
               [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
               [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
               [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
               [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
               [102, 255, 0], [92, 0, 255]]
    return cmap


def get_iacl_colormap():
    cmap = [[0, 0, 127],
            [0, 0, 254],
            [0, 96, 256],
            [0, 212, 255],
            [76, 255, 170],
            [170, 255, 76],
            [255, 229, 0],
            [255, 122, 0],
            [254, 18, 0]]
    return cmap

def get_retouch_colormap():
    cmap = [[0, 0, 0],
            [0, 0, 254],
            [0, 96, 256],
            [0, 212, 255],
            [76, 255, 170],
            [170, 255, 76],
            [255, 229, 0],
            [255, 122, 0],
            [254, 18, 0]]
    return cmap



DEFAULT_VALUES = {
    'sliding_miou_kernel': 7,  # Make sure this is odd!
    'sliding_miou_stride': 4,
}

DEFAULT_CONFIG_DICT = {
    'mode': 'training',
    'debugging': False,
    'log_every_n_epochs': 100,
    'max_valid_imgs': 10,
    'cuda': True,
    'gpu_device': 0,
    'parallel': False,
    'parallel_gpu_devices': [],
    'seed': 0,
    'tta': False
}

DEFAULT_CONFIG_NESTED_DICT = {
    'data': {
        'transforms': ['pad'],
        'transform_values': {
            'crop_size': 0.5,
            'crop_mode': 'random',
            'crop_shape': [512, 1024]
        },
        'split': 1,
        'batch_size': 10,
        'num_workers': 0,
        'preload': False,
        'blacklist': True,
        'use_propagated': False,
        'propagated_video_blacklist': False,
        'propagated_quart_blacklist': False,
        'use_relabeled': False,
        'weighted_random': [0, 0],
        'weighted_random_mode': 'v1',
        'oversampling': [0, 0],
        'oversampling_frac': 0.2,
        'oversampling_preset': 'default',
        'adaptive_batching': [0, 0],
        'adaptive_sel_size': 10,
        'adaptive_iou_update': 1,
        "repeat_factor": [0, 0],
        "repeat_factor_freq_thresh": 0.15,
        # loaders for two-step pseudo training
        # only loads labelled data with RF
        "lab_repeat_factor": [0, 0],
        # only loads unlabelled data
        "ulab_default": [0, 0],
        # loads lab and ulab mixed -- default choice for pseudo training
        "mixed_default": [0, 0],
        # loads lab with RF and ulab mixed
        "mixed_repeat_factor": [0, 0]
    },
    'train': {
        'epochs': 50,
        'lr_fct': 'exponential',
        'lr_batchwise': False,
        'lr_restarts': [],
        'lr_restart_vals': 1,
        'lr_params': None,
    },
    'loss': {
        'temperature': 0.1,
        'dominant_mode': 'all',
        'label_scaling_mode': 'avg_pool',
        'dc_weightings': {
            'outer_freq': False,
            'outer_entropy': False,
            'outer_confusionmatrix': False,
            'inner_crossentropy': False,
            'inner_idealcrossentropy': False,
            'neg_confusionmatrix': False,
            'neg_negativity': False
        },
    }
}
