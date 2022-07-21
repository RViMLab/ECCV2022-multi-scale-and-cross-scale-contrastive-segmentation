import pathlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Sampler
from utils import DATASETS_INFO, get_class_info, reverse_one_to_many_mapping
from itertools import islice
from torch.utils.data.distributed import DistributedSampler
from .distributed import is_distributed, get_rank, get_world_size
import math


def get_class_repeat_factors_for_experiment(lbl_df: pd.DataFrame,
                                            repeat_thresh: float,
                                            exp: int,
                                            return_frequencies=False,
                                            dataset: str = 'CADIS'):

    experiment_cls = DATASETS_INFO[dataset].CLASS_INFO[exp][1]
    exp_mapping = DATASETS_INFO[dataset].CLASS_INFO[exp][0]
    rev_mapping = reverse_one_to_many_mapping(exp_mapping)
    canonical_cls = DATASETS_INFO[dataset].CLASS_NAMES[0]
    canonical_num_to_name = reverse_one_to_many_mapping(DATASETS_INFO[dataset].CLASS_INFO[0][1])
    num_frames = lbl_df.shape[0]

    cls_freqs = dict()
    cls_rfs = dict()

    for c in canonical_cls:
        c_exp = rev_mapping[canonical_num_to_name[c]]  # from canonical cls name to experiment num
        if c_exp not in cls_freqs.keys():
            cls_freqs[c_exp] = 0
        s = lbl_df.loc[lbl_df[c] > 0].shape[0]
        cls_freqs[c_exp] += s / num_frames

    for c_exp in experiment_cls:
        if cls_freqs[c_exp] == 0:
            cls_freqs[c_exp] = repeat_thresh
        cls_rfs[c_exp] = np.maximum(1, np.sqrt(repeat_thresh / cls_freqs[c_exp]))
    cls_freqs = {k: v for k, v in sorted(cls_freqs.items(), reverse=True, key=lambda item: item[1])}
    cls_rfs = {k: v for k, v in sorted(cls_rfs.items(), reverse=True, key=lambda item: item[1])}
    if return_frequencies:
        return cls_freqs, cls_rfs
    else:
        return cls_rfs


def get_image_repeat_factors_for_experiment(lbl_df: pd.DataFrame, cls_rfs: dict, exp: int, dataset: str):
    exp_mapping = DATASETS_INFO[dataset].CLASS_INFO[exp][0]
    rev_mapping = reverse_one_to_many_mapping(exp_mapping)  # from canonical to experiment classes
    canonical_cls = DATASETS_INFO[dataset].CLASS_NAMES[0]
    canonical_num_to_name = reverse_one_to_many_mapping(DATASETS_INFO[dataset].CLASS_INFO[0][1])  # canonical class to num
    img_rfs = []
    inds = []
    for idx, row in lbl_df.iterrows():  # for each frame
        class_repeat_factors_in_frame = []
        for c in canonical_cls:
            if row[c] > 0:
                class_repeat_factors_in_frame.append(cls_rfs[rev_mapping[canonical_num_to_name[c]]])
        img_rfs.append(np.max(class_repeat_factors_in_frame))
        inds.append(idx)
    return inds, img_rfs


class RepeatFactorSampler(Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, dataframe: pd.DataFrame,
                 repeat_thresh: float, experiment: int, split: int, blacklist=True, seed=None, dataset='CADIS'):
        """ Computes repeat factors and returns repeat factor sampler
        Note: this sampler always uses shuffling
        :param data_source: a torch dataset object
        :param dataframe: a dataframe with class occurences as columns
        :param repeat_thresh: repeat factor threshold (intuitively: frequency below which rf kicks in)
        :param experiment: experiment id
        :param split: dataset split being used to determine repeat factors for each image in it.
        :param blacklist: whether blackslisting is to be applied
        :param seed: seeding for torch randomization
        :param dataset : todo does not support CTS currently
        :return RepeatFactorSampler object
        """
        super().__init__(data_source=data_source)
        assert(0 <= repeat_thresh < 1 and split in [0, 1, 2])
        seed = 1 if seed is None else seed
        self.seed = int(seed)
        self.shuffle = True  # shuffling is always used with this sampler
        self.split = split
        self.repeat_thresh = repeat_thresh
        df = get_class_info(dataframe, 0, with_name=True)
        if blacklist:  # drop blacklisted
            df = df.drop(df[df['blacklisted'] == 1].index)
            df.reset_index()
        self.class_repeat_factors, self.repeat_factors = \
            self.repeat_factors_class_and_image_level(df, experiment, repeat_thresh, split, dataset)
        self._int_part = torch.trunc(self.repeat_factors)
        self._frac_part = self.repeat_factors - self._int_part
        self.g = torch.Generator()
        self.g.manual_seed(self.seed)
        self.epoch = 0
        self.indices = None
        self.distributed = is_distributed() # todo this should be removed in the future once local has ddp package

        self.num_replicas = get_world_size()
        self.rank = get_rank()
        print(f'RF sampler -- world_size: {self.num_replicas} rank : {self.rank}')
        self.dataset = data_source
        # if len(self.dataset) % self.num_replicas ==0: # type: ignore
        #     self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        # else:
        #     self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        # self.total_size = self.num_samples * self.num_replicas

    @staticmethod
    def repeat_factors_class_and_image_level(df: pd.DataFrame, experiment: int, repeat_thresh: float,
                                             split: int, dataset: str):
        train_videos = DATASETS_INFO[dataset].DATA_SPLITS[split][0]
        train_df = df.loc[df['vid_num'].isin(train_videos)]
        train_df = train_df.reset_index()
        # For each class compute the class-level repeat factor: r(c) = max(1, sqrt(t/f(c)) where f(c) is class freq
        class_rfs = get_class_repeat_factors_for_experiment(train_df, repeat_thresh, experiment, dataset=dataset)
        # For each image I, compute the image-level repeat factor: r(I) = max_{c in I} r(c)
        inds, rfs = get_image_repeat_factors_for_experiment(train_df, class_rfs, experiment, dataset)
        return class_rfs, torch.tensor(rfs, dtype=torch.float32)

    def __iter__(self):
        if self.distributed: # todo this should be removed in the future
            start = get_rank()
            step = get_world_size() # 1 if not ddp
            # to debug
            # print(f'rank {get_rank()} -slicing start {start} step {step} ')
            print(f'rank {get_rank()} indices : {len([i for i in islice(self._yield_indices(), start, None, step)])}')
            yield from islice(self._yield_indices(), start, None, step)
        else:

            yield from islice(self._yield_indices(), 0, None, 1)

    def _yield_indices(self):
        if self.indices is not None:
            indices = torch.tensor(self.indices, dtype=torch.int64)
        else:
            indices = self._get_epoch_indices(self.g)
        ind_left = self.__len__()
        print(f'Indices generated {ind_left}, rank : {get_rank()}')
        self.g.manual_seed(self.seed + self.epoch)
        while ind_left > 0:
            # each epoch may have a slightly different size due to the stochastic rounding.
            randperm = torch.randperm(len(indices), generator=self.g)  # shuffling
            for item in indices[randperm]:
                # print(f'yielding : {item} rank : {get_rank()}')
                yield int(item)
                ind_left -= 1
        self.indices = None

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self._get_epoch_indices(self.g))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_epoch_indices(self, generator):
        # stochastic rounding so that the target repeat factor
        # is achieved in expectation over the course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rounded_rep_factors = self._int_part + (rands < self._frac_part).float()
        indices = []
        # replicate each image's index by its rounded repeat factor
        for img_index, rep_factor in enumerate(rounded_rep_factors):
            indices.extend([img_index] * int(rep_factor.item()))
        self.indices = indices
        if  self.num_replicas>1: # self.distributed and
            # ensures each process has access to equal number of indices from the dataset
            self.num_indices = len(self.indices)
            if self.num_indices % self.num_replicas ==0:
                self.indices_per_processs =  math.ceil(self.num_indices / self.num_replicas)
            else:
                self.indices_per_processs = math.ceil((self.num_indices - self.num_replicas) / self.num_replicas)

            self.num_indices_to_keep = self.indices_per_processs * self.num_replicas
            self.indices_to_keep = torch.randint(low=0, high=self.num_indices_to_keep-1,
                                            size=[self.num_indices_to_keep],
                                            generator=generator)

            # print(f'num_indices = {self.num_indices} - num_indices_to_keep = {self.self.num_indices_to_keep} - rank : {get_rank()}' )
            return torch.tensor(indices, dtype=torch.int64)[self.indices_to_keep]

        return torch.tensor(indices, dtype=torch.int64)



if __name__ == '__main__':
    inds = np.arange(1000).tolist()
    def dummy(start):
        yield from islice(inds, start, None, 4)
    a = [[i for i in dummy(start)] for start in range(4)]
