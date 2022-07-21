import json
import os
from collections import namedtuple
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
from PIL import Image, ImageFile
from utils import DATASETS_INFO, remap_mask, printlog
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pathlib


class Cityscapes(Dataset):
    """`Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="fine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): The quality mode to use, ``fine`` or ``coarse``
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    Examples:

        Get semantic segmentation target

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type='semantic')

            img, smnt = dataset[0]

        Get multiple targets

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='train', mode='fine',
                                 target_type=['instance', 'color', 'polygon'])

            img, (inst, col, poly) = dataset[0]

        Validate on the "coarse" set

        .. code-block:: python

            dataset = Cityscapes('./data/cityscapes', split='val', mode='coarse',
                                 target_type='semantic')

            img, smnt = dataset[0]
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(self, root, transforms_dict, split='train', mode='fine', target_type='semantic', debug=False):
        """

        :param root: path to cityscapes dir (i.e where directories "leftImg8bit" and "gtFine" are located)
        :param transforms_dict: see dataset_from_df.py
        :param split: any of "train", "test", "val"
        :param mode: if "fine" then loads finely annotated images else Coarsely uses coarsely annotated
        :param target_type: currently only expects the default: 'semantic' (todo: test other target_types if needed)
        """
        super(Cityscapes, self).__init__()
        self.root = root
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.lbl_transforms = Compose(transforms_dict['lbl'])
        assert(mode in ("fine", "coarse"))
        valid_modes = ["train", "test", "val", ['train', 'val']] if mode == "fine" else ("train", "train_extra", "val")
        assert (split in valid_modes), f'split {split} is not in valid_modes {valid_modes}'
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.split = split  # "train", "test", "val"
        self.debug = debug
        self.target_type = target_type
        self.images = []
        self.targets = []
        # this can only take the following values so hardcoded
        self.dataset = 'CITYSCAPES'
        self.experiment = 1

        if self.split == ['train', 'val']:
            # for training on train + val
            printlog('train set is train+val splits')
            self.images_dir = []
            self.targets_dir = []
            for s in self.split:
                self.images_dir += [os.path.join(self.root, 'leftImg8bit', s)]
                self.targets_dir += [os.path.join(self.root, self.mode, s)]

            if not isinstance(target_type, list):
                self.target_type = [target_type]
            for value in self.target_type:
                # assert(value in ["instance", "semantic", "polygon", "color"])
                assert(value in ["semantic"])
            for images_dir, targets_dir in zip(self.images_dir, self.targets_dir):
                for city in os.listdir(images_dir):
                    img_dir = os.path.join(images_dir, city)
                    target_dir = os.path.join(targets_dir, city)
                    for file_name in os.listdir(img_dir):
                        target_types = []
                        for t in self.target_type:
                            target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                         self._get_target_suffix(self.mode, t))
                            target_types.append(os.path.join(target_dir, target_name))

                        self.images.append(os.path.join(img_dir, file_name))
                        self.targets.append(target_types)
                        assert(pathlib.Path(self.images[-1]).exists() and pathlib.Path(*self.targets[-1]).exists())
            printlog(f'cts data all found split = {self.split}')
            a = 1
        else:
            self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
            self.targets_dir = os.path.join(self.root, self.mode, split)

            if not isinstance(target_type, list):
                self.target_type = [target_type]
            for value in self.target_type:
                # assert(value in ["instance", "semantic", "polygon", "color"])
                assert(value in ["semantic"])

            for city in os.listdir(self.images_dir):
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)
                for file_name in os.listdir(img_dir):
                    target_types = []
                    for t in self.target_type:
                        target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                     self._get_target_suffix(self.mode, t))
                        target_types.append(os.path.join(target_dir, target_name))

                    self.images.append(os.path.join(img_dir, file_name))
                    self.targets.append(target_types)

        self.return_filename = False

    def __getitem__(self, index, ):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                # todo: this possibly leads to errors - only semantic is used for now
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]
        metadata = {'index': index}
        target = remap_mask(np.array(target),
                            DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0], to_network=True).astype('int32')
        # print(index, ': ', np.unique(target))
        target = Image.fromarray(target)
        # if 14 in np.unique(target).tolist():  # tracl
        #     target.show()
        #     target.close()
        # return 0 , 0 , 0
        image, target, metadata = self.common_transforms((image, target, metadata))
        img_tensor = self.img_transforms(image)
        lbl_tensor = self.lbl_transforms(target).squeeze()

        if self.return_filename:
            metadata.update({'img_filename': self.images[index],
                             'target_filename': self.targets[index][0]})

        if self.debug:
            ToPILImage()(img_tensor).show()
            ToPILImage()(lbl_tensor).show()
            print(f'\nafter aug index, : {np.unique(lbl_tensor)}  lbl {lbl_tensor.shape} image {img_tensor.shape} fname:{self.images[index]}')


        return img_tensor, lbl_tensor, metadata

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    @staticmethod
    def _load_json(path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    @staticmethod
    def _get_target_suffix(mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)


# if __name__ == '__main__':
#     import pathlib
#     data_path = r'C:\Users\Theodoros Pissas\Documents\tresorit\CITYSCAPES/'
#     from torchvision.transforms import ToTensor
#     train_set = Cityscapes(root=data_path, split='val', mode='fine', target_type='semantic',
#                            transforms_dict={'common': [], 'img': [], 'lbl': [(ToTensor())]})
#
#     train_set.return_filename = True
#     for i in range(100):
#         ret = train_set[i]
#         print(ret)
#     a = 1

if __name__ == '__main__':
    import pathlib
    import torch
    from utils import parse_transform_lists
    import json
    import cv2
    from torch.nn import functional as F
    from utils import Pad, RandomResize, RandomCropImgLbl, Resize, FlipNP, to_numpy, pil_plot_tensor, to_comb_image
    from torchvision.transforms import ToTensor
    import PIL.Image as Image

    data_path = r'C:\Users\Theodoros Pissas\Documents\tresorit\CITYSCAPES/'
    d = {"dataset":'CITYSCAPES', "experiment":1}
    path_to_config = '../configs/hrnet_contrastive_CTS.json'
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']
    if 'torchvision_normalise' in transforms_list:
        del transforms_list[-1]

    transforms_dict = parse_transform_lists(transforms_list, transforms_values, **d)



    transforms_list_val = config['data']['transforms_val']
    transforms_values_val = config['data']['transform_values_val']
    transforms_dict_val = parse_transform_lists(transforms_list_val, transforms_values_val, **d)

    train_set = Cityscapes(root=data_path,
                           debug=True,
                           split='train',
                           mode = 'fine',
                           target_type='semantic',
                           transforms_dict=transforms_dict)
    valid_set = Cityscapes(root=data_path,
                           debug=True,
                           split='val',
                           mode='fine',
                           target_type='semantic',
                           transforms_dict=transforms_dict_val)

    issues = []
    train_set.return_filename = True
    hs=[]
    ws = []
    for ret in train_set:
        # print(ret[0].shape)
        # img = ToPILImage()(ret[0]).show()
        # lbl = ToPILImage()(ret[1]).show()

        hs.append(ret[0].shape[1])
        ws.append(ret[0].shape[2])
        # print(ret[-1])
        # print('*'*10)
        # meta = ret[-1]
        # lbl = meta['original_labels'].unsqueeze(0)
        # resized = ret[1].unsqueeze(0).unsqueeze(0).long()
        # pad_w, pad_h, stride = meta["pw_ph_stride"]
        # if pad_h > 0 or pad_w > 0:
        #
        #     un_padded = resized[:, :, 0:resized.size(2) - pad_h, 0:resized.size(3) - pad_w]
        #     pil_plot_tensor(un_padded)
        #     un_resized = F.interpolate(un_padded.float(), size=lbl.size()[-2:], mode='nearest')
        #     print(torch.sum(un_resized- lbl))


        present_classes = torch.unique(ret[1])
        if 14 in present_classes:
            issues.append([ret[-1], present_classes])
            print('truck found !!!! ')
            print(present_classes)
            # pil_plot_tensor(ret[0], is_rgb=True)
            # pil_plot_tensor(ret[1], is_rgb=False)
            #
            cv2.imshow('truck', to_comb_image(ret[0], ret[1], None, 1, 'CITYSCAPES'))
            a = 1
        # elif 15 in present_classes:
        #     issues.append([ret[-1], present_classes])
        #     print('bus found !!!! ')
        #     print(present_classes)
        #     pil_plot_tensor(ret[0], is_rgb=True)
        #     pil_plot_tensor(ret[1], is_rgb=False)

        a = 1
    print(max(hs))
    print(max(ws))


classes_exp1 = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
    255: 'Ignore'
}