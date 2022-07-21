import numpy as np
from PIL import Image, ImageFilter
import random
from utils.defaults import DATASETS_INFO
from torchvision.transforms import RandomCrop, ToTensor
from torchvision.transforms.functional import crop
from typing import Union
import math
from utils.logger import printlog


class BaseTranform:
    def __init__(self,
                 dataset,
                 experiment,
                 img_pad_value:float=0.0):

        self.dataset = dataset
        self.experiment = experiment
        self.ignore_class = (len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]) - 1) \
            if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] else -1
        self.n_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1])
        self.img_pad_value = img_pad_value
        self.label_pad_value = self.ignore_class
    def __call__(self, **kwargs):
        pass


class BlurPIL(BaseTranform):
    """PIL-based function to randomly blur img"""
    def __init__(self,
                 dataset:str,
                 experiment:int,
                 probability: float,
                 kernel_limits: tuple):

        super().__init__(dataset, experiment)
        self.probability = probability
        self.kernel_limits = kernel_limits

    def __call__(self, img: Image):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if np.random.random() < self.probability:
            img = img.filter(ImageFilter.GaussianBlur(radius=np.random.randint(*self.kernel_limits)))
        return np.array(img)


class RandomCropImgLbl(BaseTranform):
    """PIL and torchvision based function to randomly crop img and label"""
    def __init__(self,
                 dataset:str,
                 experiment:int,
                 shape: Union[tuple, list],
                 crop_class_max_ratio:Union[float,None]=None):

        super().__init__(dataset, experiment)
        self.crop_shape = shape
        self.crop_class_max_ratio = crop_class_max_ratio # max ratio a class can occupy over
        self.random_cropper = RandomCrop(size=shape) # from torchvision accepts PIL or tensor
        self.patience = 10 # number of attempts for enforcing crop class max ratio

    def __call__(self, arrs) -> tuple:
        # def __call__(self, img, lbl, metadata=None) -> tuple:
        img = arrs[0]
        lbl = arrs[1]
        metadata = arrs[2] if len(arrs) == 3 else None
        if not isinstance(img, Image.Image) or not isinstance(lbl, Image.Image):
            img = Image.fromarray(img)
            lbl = Image.fromarray(lbl)

        cls_distr = None # metadata only
        found = False
        if self.crop_class_max_ratio:
            # attempts to find a crop where the dominant class has no more that crop_class_max_ratio frequency
            for _ in range(self.patience):
                i, j, h, w = self.random_cropper.get_params(img, self.crop_shape)
                lbl_crop = crop(lbl, i, j, h, w)
                classes, cnt = np.unique(np.asarray(lbl_crop), return_counts=True)
                cnt = cnt[classes != self.ignore_class]
                cls_distr = cnt / np.sum(cnt)
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.crop_class_max_ratio:
                    img = crop(img, i, j, h, w)
                    lbl = lbl_crop.copy()
                    found = True
                    break
            if not found:
                # print('found', found)
                lbl = crop(lbl, i, j, h, w)
                img = crop(img, i, j, h, w)

        else:
            i, j, h, w = self.random_cropper.get_params(img, self.crop_shape)
            # img = crop(img, i, j, h, w)
            lbl = crop(lbl, i, j, h, w)
            img = crop(img, i, j, h, w)

        if metadata:
            metadata['crop_ijhw'] = [i, j, h, w]
            # if cls_distr is not None:
            #     # collate complains so must fix size of this to list of n_classes elements
            #     metadata['cls_distr'] = [0]*self.n_classes
            #     for ind, freq in enumerate(cls_distr):
            #         cl = classes.tolist()[ind]
            #         metadata['cls_distr'][cl] = freq
            return np.array(img), np.array(lbl), metadata
        else:
            return np.array(img), np.array(lbl)

class Resize(BaseTranform):
    def __init__(self,
                 dataset, experiment,
                 target_size:Union[list, tuple, None]=None,
                 min_side_length:Union[int, None]=None,
                 keep_aspect_ratio:bool=True,
                 fit_stride:Union[int, None]=8,
                 img_pad_value:float=0.0,
                 return_original_labels=False):
        # todo make ignore class dataset-dependent and equal to num_classes -1

        super().__init__(dataset, experiment, img_pad_value)
        assert ((target_size is not None) or (min_side_length is not None)),\
            'Resize() must have either a fixed target_size [h,w] or min_size_length '

        self.target_size = target_size
        if self.target_size is not None:
            assert (isinstance(target_size, list) or isinstance(target_size, tuple) and (len(target_size)==2)),\
                f'{target_size} is invalid'
            self.target_size = target_size[::-1] # for PIL: H,W --> W,H
        else:
            assert min_side_length is not None, f'target_size was [{target_size}]' \
                                                f' and min_side_length was [{min_side_length}] provide one of the two'

        # setting min_side length is equivalent to preserve aspect ratio but with min side being equal with min_side_length
        self.min_side_length = min_side_length
        self.keep_aspect_ratio = keep_aspect_ratio # currently unused
        # fit a network's output to be divisible to an integer stride by padding after resizing
        # ex (511, 251) --> stride = 32 --> (512, 256)
        self.fit_stride = fit_stride

        # return non-resized labels for testing/validation (ex. for Pascal context evaluation due to diverse sizes)
        self.return_original_labels = return_original_labels

        printlog(f'Resize: \n'
                 f'  target_size: {self.target_size}\n'
                 f'  preserve aspect ratio :{self.keep_aspect_ratio or (self.min_side_length is not None)}\n'
                 f'  fit_stride :{self.fit_stride}\n'
                 f'  return original labels : {self.return_original_labels}\n')

    def __call__(self, arrs):
        img = arrs[0]
        label = arrs[1]
        metadata = arrs[2] if len(arrs) == 3 else None
        if not isinstance(img, Image.Image) or not isinstance(label, Image.Image):
            img = Image.fromarray(img)
            label = Image.fromarray(label)

        width, height = img.size
        input_size = (width, height)
        h_scale_ratio, w_scale_ratio = 1.0, 1.0

        if self.target_size is not None:
            target_size = self.target_size
            w_scale_ratio = self.target_size[0] / width
            h_scale_ratio = self.target_size[1] / height

        elif self.min_side_length is not None:
            # preserves aspect ratio: min_side_length / min_side = scaling factor for both width and height
            scale_ratio = self.min_side_length / min(width, height)
            w_scale_ratio, h_scale_ratio = scale_ratio, scale_ratio
            target_size = [int(round(width * w_scale_ratio)),
                           int(round(height * h_scale_ratio))]

        img = img.resize(target_size, Image.BILINEAR)

        if self.return_original_labels:
            metadata['original_labels'] = ToTensor()(label.copy())

        label = label.resize(target_size, Image.NEAREST)

        if self.fit_stride:
            stride = self.fit_stride
            target_width, target_height = target_size[0], target_size[1]
            pad_cols = 0 if (target_width % stride == 0) else stride - (target_width % stride)  # right
            pad_rows = 0 if (target_height % stride == 0) else stride - (target_height % stride)  # down
            pad_img = ((0, pad_rows), (0, pad_cols), (0, 0))
            pad_label = ((0, pad_rows), (0, pad_cols))
            kwargs = {'mode': 'constant', 'constant_values': self.img_pad_value}
            img = np.pad(np.array(img), pad_width=pad_img, **kwargs)
            label = np.pad(np.array(label), pad_width=pad_label, constant_values=self.label_pad_value)

        if metadata:
            output_shape = img.shape[::-1] if isinstance(img,np.ndarray) else img.size
            metadata['sh_sw_in_out'] = (h_scale_ratio, w_scale_ratio, input_size, output_shape)
            if self.fit_stride:
                metadata['pw_ph_stride'] = (pad_cols, pad_rows, self.fit_stride)
            return np.array(img), np.array(label), metadata
        else:
            return np.array(img), np.array(label)


class RandomResize(BaseTranform):

    def __init__(self,
                 dataset:str,
                 experiment:int,
                 scale_range=(0.5, 2.0),
                 aspect_range=(0.9, 1.1),
                 probability=0.5,
                 target_size=None,
                 img_pad_value:float=0.0):
        """
        Resize the given numpy.ndarray to random size and aspect ratio
        """

        super().__init__(dataset, experiment, img_pad_value)

        # todo make ignore class dataset-dependent and equal to num_classes -1
        self.scale_range = scale_range
        self.aspect_range = aspect_range
        self.probability = probability
        self.target_size = target_size
        if target_size is not None:
            assert(isinstance(target_size, list) or isinstance(target_size, tuple) and (len(target_size)==2))
            self.target_size = target_size[::-1] # for PIL: H,W --> W,H
            self.pad = True
            self.pad_mode = 'constant'
        else:
            self.pad = False
            self.pad_mode = None

        printlog(f'RandomResize: \n'
                 f'  scale_range: {self.scale_range} - aspect_range {self.aspect_range}\n'
                 f'  probability: {self.probability}\n'
                 f'  target_size: {self.target_size}\n'
                 f'  padding: {self.pad} : pad values im/lbl {self.img_pad_value}, {self.label_pad_value}')

    def _get_scale(self):
        # get scale factor uniformely at random in scale_range interval
        return random.uniform(self.scale_range[0], self.scale_range[1])

    def __call__(self, arrs):
        img = arrs[0]
        label = arrs[1]
        metadata = arrs[2] if len(arrs) == 3 else None
        if not isinstance(img, Image.Image) or not isinstance(label, Image.Image):
            img = Image.fromarray(img)
            label = Image.fromarray(label)

        w_scale_ratio, h_scale_ratio = 1.0, 1.0
        width, height = img.size
        if np.random.random() < self.probability:
            scale_factor = self._get_scale() # get a random scale factor
            aspect_ratio = random.uniform(*self.aspect_range)
            w_scale_ratio = math.sqrt(aspect_ratio) * scale_factor
            h_scale_ratio = math.sqrt(1.0 / aspect_ratio) * scale_factor

            new_size = (int(width * w_scale_ratio), int(height * h_scale_ratio))
            img = img.resize(new_size, Image.BILINEAR)
            label = label.resize(new_size, Image.NEAREST)

            if self.pad:
                pad_width = self.target_size[0] - new_size[0]  # check if width is less than target (i.e crop size)
                pad_height = self.target_size[1] - new_size[1]  # check if height is less than target (i.e crop size)
                if pad_width > 0 or pad_height > 0:
                    col_pad = random.randint(0, pad_width) if pad_width>0 else 0  # pad_left
                    row_pad = random.randint(0, pad_height)  if pad_height>0 else 0# pad_up
                    pad_img = ((row_pad, max(0,pad_height-row_pad)), (col_pad, max(0, pad_width-col_pad)),  (0, 0))
                    pad_label = ((row_pad, max(0,pad_height-row_pad)), (col_pad, max(0, pad_width-col_pad)))
                    kwargs ={'mode':self.pad_mode, 'constant_values':self.img_pad_value}
                    img = np.pad(np.array(img), pad_width=pad_img, **kwargs)
                    label = np.pad(np.array(label), pad_width=pad_label, constant_values=self.label_pad_value)
                    assert(np.array(label).shape[::-1] == np.array(img).shape[0:2][::-1])

        else:
            if metadata:
                metadata['random_scale_ratio_hw'] = (1, 1)

        if metadata:
            metadata['sh_sw_in_out'] = (h_scale_ratio, w_scale_ratio, (height, width), np.array(label).shape[::-1])
            return np.array(img), np.array(label), metadata
        else:
            return np.array(img), np.array(label)



class Pad(BaseTranform):
    """
    random padding to make size fixed target_size
    """
    def __init__(self,
                 dataset:str,
                 experiment:int,
                 target_size=None,
                 fit_stride:Union[None, int]=None,
                 img_pad_value:float=0.0):
        super().__init__(dataset, experiment, img_pad_value)

        self.target_size = target_size
        self.pad_mode = 'constant'
        self.img_pad_value = img_pad_value
        self.fit_stride = fit_stride

    def __call__(self, arrs):
        img = arrs[0]
        label = arrs[1]
        metadata = arrs[2] if len(arrs) == 3 else None
        if not isinstance(img, Image.Image) or not isinstance(label, Image.Image):
            img = Image.fromarray(img)
            label = Image.fromarray(label)

        input_size = img.size

        pad_width = self.target_size[0] - input_size[0]  # check if width is less than target (i.e crop size)
        pad_height = self.target_size[1] - input_size[1]  # check if height is less than target (i.e crop size)
        if pad_width > 0 or pad_height > 0:
            col_pad = random.randint(0, pad_width) if pad_width > 0 else 0  # pad_left
            row_pad = random.randint(0, pad_height) if pad_height > 0 else 0  # pad_up
            pad = ((row_pad, max(0, pad_height - row_pad)), (col_pad, max(0, pad_width - col_pad)), (0, 0))
            pad_label = ((row_pad, max(0, pad_height - row_pad)), (col_pad, max(0, pad_width - col_pad)))
            pad_val_img = self.img_pad_value
            pad_val_label = self.label_pad_value
            # print(f'pad {pad}')
            kwargs = {'mode': 'constant', 'constant_values': pad_val_img} if self.pad_mode == 'constant' \
                else {'mode': 'reflect'}
            img = np.pad(np.array(img), pad_width=pad, **kwargs)
            label = np.pad(np.array(label), pad_width=pad_label, constant_values=pad_val_label)
            # print(f'padded size: {img.shape}')
            # Image.fromarray(img).show()
            # Image.fromarray(label).show()

        if metadata:
            output_shape = img.shape[::-1] if isinstance(img, np.ndarray) else img.size
            metadata['random_pad-w_pad-h_size'] = (pad_width, pad_height, output_shape)
            return np.array(img), np.array(label), metadata
        else:
            return np.array(img), np.array(label)

