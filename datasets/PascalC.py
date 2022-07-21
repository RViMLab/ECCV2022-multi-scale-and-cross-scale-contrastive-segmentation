import json
import os
import torch
from collections import namedtuple
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
from PIL import Image, ImageFile
from utils import DATASETS_INFO, remap_mask, printlog, mask_to_colormap, get_remapped_colormap
import numpy as np
# import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pathlib


class PascalC(Dataset):
    def __init__(self, root, transforms_dict, split='train', mode='fine', target_type='semantic', debug=False):
        """
        :param root: path to pascal dir (i.e where directories "leftImg8bit" and "gtFine" are located)
        :param transforms_dict: see dataset_from_df.py
        :param split:  "train" or "val"
        :param mode: if "fine" then loads finely annotated images else Coarsely uses coarsely annotated
        :param target_type: currently only expects the default: 'semantic' (todo: test other target_types if needed)
        """
        self.debug = debug
        super(PascalC, self).__init__()
        self.root = root
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.lbl_transforms = Compose(transforms_dict['lbl'])
        valid_modes = ["train", "val"]
        assert (split in valid_modes), f'split {split} is not in valid_modes {valid_modes}'
        # self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.split = split  # "train", "test", "val"

        # self.target_type = target_type
        self.images = []
        self.targets = []
        # this can only take the following values so hardcoded
        self.dataset = 'PASCALC'
        self.experiment = 1

        # for training on train + val
        self.images_dir = []
        self.targets_dir = []
        self.images_dir = pathlib.Path(os.path.join(self.root, self.split, 'image'))
        self.targets_dir = pathlib.Path(os.path.join(self.root, self.split, 'label'))

        for img_path, target_path in zip(sorted(self.images_dir.glob('*.jpg')), sorted(self.targets_dir.glob('*.png'))):
            self.images.append(img_path)
            self.targets.append(target_path)
            assert(pathlib.Path(self.images[-1]).exists() and pathlib.Path(self.targets[-1]).exists())
            assert(pathlib.Path(self.images[-1]).stem == pathlib.Path(self.targets[-1]).stem)
        printlog(f'{self.dataset} data all found split is [ {self.split} ]')

        self.return_filename = False

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        # if self.debug:
        #     image.show()
        # target.show()


        target = remap_mask(np.array(target),
                            DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0], to_network=True).astype('int32')

        target = Image.fromarray(target)

        # print(index, ': ', np.unique(target), '  ', [class_int_to_name[c] for c in np.unique(target) if not (c==59) ])
        # class_int_to_name = DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]

        metadata = {'index': index}
        image, target, metadata = self.common_transforms((image, target, metadata))
        img_tensor = self.img_transforms(image)
        lbl_tensor = self.lbl_transforms(target).squeeze()

        if self.return_filename:
            metadata.update({'img_filename': str(self.images[index]),
                             'target_filename': str(self.targets[index])})
        if self.debug:
            ToPILImage()(img_tensor).show()
            ToPILImage()(lbl_tensor).show()
            #
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
            # p = pathlib.Path(r'C:\Users\Theodoros Pissas\Documents\tresorit\PASCALC\visuals\val/')
            # p1 = pathlib.Path(f'{fn}')
            # # ToPILImage()(lbl_tensor).save(f"{str(p/p1)}")
            #
            # cv2.imwrite(f"{str(p/p1)}", debug_lbl)
            #
            print(f'\nafter aug index, : {np.unique(lbl_tensor)}  lbl {lbl_tensor.shape} image {img_tensor.shape} fname:{self.images[index]}')

        return img_tensor, lbl_tensor, metadata

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)


if __name__ == '__main__':
    import pathlib
    from torch.nn import functional as F
    from utils import Pad, RandomResize, RandomCropImgLbl, Resize, FlipNP, to_numpy, pil_plot_tensor
    data_path = r'C:\Users\Theodoros Pissas\Documents\tresorit\PASCALC/'
    from torchvision.transforms import ToTensor
    import PIL.Image as Image
    d = {"dataset":'PASCALC', "experiment":1}

    # augs= [
    #        FlipNP(probability=(0, 1.0)),
    #        RandomResize(**d,
    #                     scale_range=[0.5, 2],
    #                     aspect_range=[0.9, 1.1],
    #                     target_size=[520, 520],
    #                     probability=1.0),
    #        RandomCropImgLbl(**d,
    #                         shape=[512, 512],
    #                         crop_class_max_ratio=0.75),
    #        ]
    #
    # augs_val= [Resize(**d, min_side_length=512, fit_stride=32, return_original_labels=True)]
    #
    #
    # train_set = PascalC(root=data_path, debug=True,
    #                     split='train',
    #                     transforms_dict={'common': augs_val,
    #                                      'img': [(ToTensor())],
    #                                      'lbl': [(ToTensor())]})

    from utils import parse_transform_lists
    import json
    path_to_config = '../configs/PASCALC/hrnet_contrastive_PC.json'
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    transforms_list = config['data']['transforms']
    if 'torchvision_normalise' in transforms_list:
        del transforms_list[-1]
    transforms_values = config['data']['transform_values']
    transforms_dict = parse_transform_lists(transforms_list, transforms_values, dataset='PASCALC', experiment=1)

    transforms_list_val = config['data']['transforms_val']
    if 'torchvision_normalise' in transforms_list:
        del transforms_list[-1]

    transforms_values_val = config['data']['transform_values_val']
    transforms_dict_val = parse_transform_lists(transforms_list_val, transforms_values_val, dataset='PASCALC', experiment=1)

    # transforms_values_val = {}
    # transforms_dict_val = parse_transform_lists({}, transforms_values_val, dataset='PASCALC', experiment=1)


    train_set = PascalC(root=data_path,
                        debug=True,
                        split='train',
                        transforms_dict=transforms_dict)

    valid_set = PascalC(root=data_path,
                        debug=True,
                        split='val',
                        transforms_dict=transforms_dict_val)
    valid_set.return_filename = True

    issues = []
    train_set.return_filename = True
    hs=[]
    ws = []
    for ret in valid_set:
        # print(ret[0].shape)
        # img = ToPILImage()(ret[0]).show()
        # lbl = ToPILImage()(ret[1]).show()

        hs.append(ret[0].shape[1])
        ws.append(ret[0].shape[2])
        print(ret[-1])
        print('*'*10)
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
        if len(present_classes) == 1 and 59 in present_classes:
            issues.append([ret[-1], present_classes])
            print('issue found !!!! ')
            print(present_classes, ret[-1])
            print('issue found !!!! ')
        a = 1
    print(max(hs))
    print(max(ws))