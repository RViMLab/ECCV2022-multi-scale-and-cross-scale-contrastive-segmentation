import os
import pathlib
import cv2
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, ToPILImage
import torch
from utils import DATASETS_INFO, remap_mask
import numpy as np


class DatasetFromDF(Dataset):
    def __init__(self, dataframe, experiment, transforms_dict, data_path=None, labels_remaped=False,
                 return_pseudo_property=False, dataset='CADIS', debug=False):
        self.df = dataframe
        self.experiment = experiment
        self.dataset = dataset
        self.common_transforms = Compose(transforms_dict['common'])
        self.img_transforms = Compose(transforms_dict['img'])
        self.lbl_transforms = Compose(transforms_dict['lbl'])
        self.labels_are_remapped = labels_remaped  # used when reading pseudo labeled data
        self.return_pseudo_property = return_pseudo_property  # used to return whether the datapoint is pseudo labelled
        self.preloaded = False if data_path is not None else True
        if self.preloaded:  # Data preloaded, need to assert that 'image' and 'label' exist in the dataframe
            assert 'image' in self.df and 'label' in self.df, "For preloaded data, the dataframe passed to the " \
                                                              "PyTorch dataset needs to contain the columns 'image' " \
                                                              "and 'label'"
        else:  # Standard case: data not preloaded, needs base path to get images / labels from
            assert 'img_path' in self.df and 'lbl_path' in self.df, "The dataframe passed to the PyTorch dataset needs"\
                                                                    " to contain the columns 'img_path' and 'lbl_path'"
            self.data_path = data_path
        self.debug = debug

    def __getitem__(self, item):
        if self.preloaded:
            img = self.df.iloc[item].loc['image']
            lbl = self.df.iloc[item].loc['label']
        else:
            # img = cv2.imread(str(pathlib.Path(self.data_path) / self.df.iloc[item].loc['img_path']))[..., ::-1]
            img = cv2.imread(
                os.path.join(
                    self.data_path,
                    os.path.join(*self.df.iloc[item].loc['img_path'].split('\\'))))[..., ::-1]
            img = img - np.zeros_like(img)  # deals with negative stride error
            # lbl = cv2.imread(str(pathlib.Path(self.data_path) / self.df.iloc[item].loc['lbl_path']), 0)
            lbl = cv2.imread(
                os.path.join(
                    self.data_path,
                    os.path.join(*self.df.iloc[item].loc['lbl_path'].split('\\'))), 0)
            lbl = lbl - np.zeros_like(lbl)

        if self.labels_are_remapped:
            # if labels are pseudo they are already remapped to experiment label set
            lbl = lbl.astype('int32')
        else:
            lbl = remap_mask(lbl, DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][0], to_network=True).astype('int32')

        # Note: .astype('i') is VERY important. If left in uint8, ToTensor() will normalise the segmentation classes!

        # Here (and before Compose(lbl_transforms) we'd need to set the random seed and pray, following this idea:
        # https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        # Big yikes. Big potential problem source, see here: https://github.com/pytorch/pytorch/issues/7068
        # If that doesn't work, the whole transforms structure needs to be changed into all-custom functions that will
        # transform both img and lbl at the same time, with one random shift / flip / whatever being applied to both
        metadata = {'index': item, 'filename': self.df.iloc[item].loc['img_path'],
                    'target_filename': str(pathlib.Path(self.df.iloc[item].loc['img_path']).stem)}

        if self.dataset == 'RETOUCH':
            subject_id = pathlib.Path(metadata['filename']).parent.stem
            slice_id = pathlib.Path(self.df.iloc[item].loc['lbl_path']).stem
            metadata['subject_id'] = subject_id
            metadata['target_filename'] = f"{subject_id}_{slice_id}"

        img, lbl, metadata = self.common_transforms((img, lbl, metadata))
        img_tensor = self.img_transforms(img)
        lbl_tensor = self.lbl_transforms(lbl).squeeze()
        if self.return_pseudo_property:
            # pseudo_tensor = torch.from_numpy(np.asarray(self.df.iloc[item].loc['pseudo']))
            metadata.update({'pseudo': self.df.iloc[item].loc['pseudo']})

        if self.debug:
            ToPILImage()(img_tensor).show()
            ToPILImage()(lbl_tensor).show()
            print(f'\nafter aug index : {np.unique(lbl_tensor)}  lbl {lbl_tensor.shape} image {img_tensor.shape}')

        return img_tensor, lbl_tensor, metadata

    def __len__(self):
        return len(self.df)
