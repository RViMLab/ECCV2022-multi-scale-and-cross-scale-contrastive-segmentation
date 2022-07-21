import json
import os
from collections import namedtuple
from typing import Union
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
from PIL import Image, ImageFile
from utils import DATASETS_INFO, remap_mask, printlog
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pathlib

from utils import DATASETS_INFO, remap_mask, printlog, mask_to_colormap, get_remapped_colormap



class ADE20K(Dataset):

    def __init__(self, root, transforms_dict, split:Union[str,list]='train', debug=False):
        """

        :param root: path to cityscapes dir (i.e where directories "leftImg8bit" and "gtFine" are located)
        :param transforms_dict: see dataset_from_df.py
        :param split: any of "train", "test", "val"
        :param mode: if "fine" then loads finely annotated images else Coarsely uses coarsely annotated
        :param target_type: currently only expects the default: 'semantic' (todo: test other target_types if needed)
        """


        super(ADE20K, self).__init__()
        self.root = root
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.lbl_transforms = Compose(transforms_dict['lbl'])
        # assert(mode in ("fine", "coarse"))
        valid_splits = ["train", "test", "val", ['train', 'val']]
        assert (split in valid_splits), f'split {split} is not in valid_modes {valid_splits}'
        # self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.split = split  # "train", "test", "val"
        self.debug = debug
        # self.target_type = target_type
        self.images = []
        self.targets = []
        # this can only take the following values so hardcoded
        self.dataset = 'ADE20K'
        self.experiment = 1
        self.img_suffix = '.jpg'
        self.target_suffix = '.png'


        if self.split == ['train', 'val']:
            # for training on train + val
            printlog('train set is train+val splits')
            for i, s in enumerate(self.split):
                self.images_dir = os.path.join(self.root, 'ADEChallengeData2016', 'images', s)
                self.targets_dir = os.path.join(self.root, 'ADEChallengeData2016', 'annotations', s)
                for image_filename in os.listdir(self.images_dir):
                    img_path = os.path.join(self.images_dir, image_filename)
                    target_path = os.path.join(self.targets_dir, image_filename.split(self.img_suffix)[-2] + self.target_suffix)
                    self.images.append(img_path)
                    self.targets.append(target_path)
                    assert (pathlib.Path(self.images[-1]).exists() and pathlib.Path(self.targets[-1]).exists())
                    assert(pathlib.Path(self.images[-1]).stem == pathlib.Path(self.targets[-1]).stem)

        elif self.split == 'test':
            self.images_dir = os.path.join(self.root, 'ADEChallengeData2016', 'images', split)
            self.targets_dir = os.path.join(self.root,'ADEChallengeData2016', 'annotations', 'train') # dummy
            targets_dummy = os.listdir(self.targets_dir)
            for n, image_filename in enumerate(os.listdir(self.images_dir)):
                img_path = os.path.join(self.images_dir, image_filename)
                target_path = os.path.join(self.targets_dir, targets_dummy[n])
                self.images.append(img_path)
                self.targets.append(target_path)
                assert (pathlib.Path(self.images[-1]).exists()) # and pathlib.Path(self.targets[-1]).exists())
                # assert(pathlib.Path(self.images[-1]).stem == pathlib.Path(self.targets[-1]).stem)
        else:
            self.images_dir = os.path.join(self.root, 'ADEChallengeData2016', 'images', split)
            self.targets_dir = os.path.join(self.root, 'ADEChallengeData2016', 'annotations', split)
            for image_filename in os.listdir(self.images_dir):
                img_path = os.path.join(self.images_dir, image_filename)
                target_path = os.path.join(self.targets_dir, image_filename.split(self.img_suffix)[-2] + self.target_suffix)
                self.images.append(img_path)
                self.targets.append(target_path)
                assert (pathlib.Path(self.images[-1]).exists() and pathlib.Path(self.targets[-1]).exists())
                assert(pathlib.Path(self.images[-1]).stem == pathlib.Path(self.targets[-1]).stem)
        printlog(f'ade20k data all found split = {self.split}, images {len(self.images)}, targets {len(self.targets)}')

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
        metadata = {'index': index}

        if self.split == 'test':
            target = remap_mask(np.ones(shape=np.array(image).shape[0:2], dtype=np.int32),
                                DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0], to_network=True).astype('int32')

        else:
            target = Image.open(self.targets[index])
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
                             'target_filename': self.targets[index]})

        if self.debug:
            # ToPILImage()(img_tensor).show()
            # ToPILImage()(lbl_tensor).show()
            # debug_lbl = mask_to_colormap(to_numpy(lbl_tensor),
            #                               get_remapped_colormap(
            #                                   DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0],
            #                                   self.dataset),
            #                               from_network=True, experiment=self.experiment,
            #                               dataset=self.dataset)[..., ::-1]
            #
            #
            #
            # fn = metadata['target_filename'].split('\\')[-1]
            # p = pathlib.Path(r'C:\Users\Theodoros Pissas\Documents\tresorit\ADEChallengeData2016\ADEChallengeData2016\visuals\val/')
            # p1 = pathlib.Path(f'{fn}')
            # # ToPILImage()(lbl_tensor).save(f"{str(p/p1)}")
            #
            # cv2.imwrite(f"{str(p/p1)}", debug_lbl)
            print(f'\nafter aug index : {np.unique(lbl_tensor)}  lbl {lbl_tensor.shape} image {img_tensor.shape} fname:{self.images[index]}')
        return img_tensor, lbl_tensor, metadata

    def __len__(self):
        return len(self.images)

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

    data_path = 'C:\\Users\\Theodoros Pissas\\Documents\\tresorit\\ADEChallengeData2016\\'
    d = {"dataset":'ADE20K', "experiment":1}
    path_to_config = '../configs/ADE20K/upnswin_contrastive_ADE20K.json'
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    transforms_list = config['data']['transforms']
    transforms_values = config['data']['transform_values']
    if 'torchvision_normalise' in transforms_list:
        del transforms_list[-1]

    transforms_dict = parse_transform_lists(transforms_list, transforms_values, **d)
    transforms_list_val = config['data']['transforms_val']
    transforms_values_val = config['data']['transform_values_val']

    if 'torchvision_normalise' in transforms_list_val:
        del transforms_list_val[-1]
    transforms_dict_val = parse_transform_lists(transforms_list_val, transforms_values_val, **d)
    del transforms_list_val[0]
    train_set = ADE20K(root=data_path,
                       debug=True,
                       split=['train', 'val'],
                       transforms_dict=transforms_dict)
    valid_set = ADE20K(root=data_path,
                       debug=True,
                       split='test',
                       transforms_dict=transforms_dict_val)

    issues = []
    valid_set.return_filename = True
    train_set.return_filename = True
    hs=[]
    ws = []
    for ret in valid_set:
        hs.append(ret[0].shape[1])
        ws.append(ret[0].shape[2])
        present_classes = torch.unique(ret[1])
        print(ret[-1])
        # elif 15 in present_classes:
        #     issues.append([ret[-1], present_classes])
        #     print('bus found !!!! ')
        #     print(present_classes)
        #     pil_plot_tensor(ret[0], is_rgb=True)
        #     pil_plot_tensor(ret[1], is_rgb=False)

        # a = 1
    # print(max(hs))
    # print(max(ws))