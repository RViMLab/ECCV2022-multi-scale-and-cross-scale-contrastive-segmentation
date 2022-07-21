import torch
from torch.nn import functional as F
from typing import Union
import datetime
import cv2
import numpy as np
from utils import printlog, to_numpy, to_comb_image, un_normalise
from models import TTAWrapper


class TTAWrapperCTS(TTAWrapper):
    def __init__(self,
                 model,
                 scale_list,
                 flip=True,
                 strides:Union[tuple, None]=None,
                 crop_size:Union[tuple, None]=None,
                 debug=False):

        super().__init__(model, scale_list, flip)
        self.num_classes = 19
        self.crop_size = crop_size if crop_size else [512,1024]
        self.strides = strides if strides else self.crop_size # defaults to no-overlapping sliding window
        self.base_size = 2048
        self.debug = debug

        printlog(f'Sliding window : strides : {self.strides} crop_size {self.crop_size}')

    def inference(self, image, flip=False, scale=1.0, id_=1):
        # image  BCHW
        assert image.device.type == 'cuda'
        size = image.size()
        pred = self.model(image)
        # done internally in model
        # pred = F.interpolate(
        #     input=pred, size=size[-2:],
        #     mode='bilinear', align_corners=self.model.align_corners
        # )
        if flip:
            flip_img = to_numpy(image)[:, :, :, ::-1]
            flip_output = self.model(torch.from_numpy(flip_img.copy()).cuda())
            # flip_output = F.interpolate(
            #     input=flip_output, size=size[-2:],
            #     mode='bilinear', align_corners=self.model.align_corners
            # )

            flip_pred = to_numpy(flip_output).copy()
            flip_pred = torch.from_numpy(flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        if self.debug:
            to_comb_image(un_normalise(image[0]), torch.argmax(pred[0], 0), None, 1, 'ADE20K', save=f'pred_scale_{scale}_{id_}.png')
        return pred.exp()

    def multi_scale_aug(self, image, label=None,
                        rand_scale=1, rand_crop=True):

        long_size = int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label = self.rand_crop(image, label)

        return image, label

    def forward(self, x):
        a = datetime.datetime.now()
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.tensor), f'x input must be a tensor instead got {type(x)}'
        batch, _, ori_height, ori_width = x.size()
        assert batch == 1, "only supporting batchsize 1."
        # x is BCHW
        image = to_numpy(x)[0].transpose((1, 2, 0)).copy()
        # x is HWC
        stride_h = int(self.strides[0] * 1.0)
        stride_w = int(self.strides[1] * 1.0)

        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()

        for scale in self.scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            # cv2.imshow(f'scale {scale}', new_img)
            height, width = new_img.shape[:-1]

            if scale < 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(new_img.cuda(), flip=True, scale=scale, id_=1)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = int(np.ceil(1.0 * (new_h - self.crop_size[0]) / stride_h)) + 1
                cols = int(np.ceil(1.0 * (new_w - self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes, new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()
                id_ = 1
                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(crop_img.cuda(), flip=self.flip, scale=scale, id_= id_)
                        id_ += 1

                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=self.model.align_corners
            )

            final_pred += preds
        if self.debug:
            to_comb_image(un_normalise(x[0]), torch.argmax(final_pred[0], 0), None, 1, 'CITYSCAPES', save=f'final.png')

        b = (datetime.datetime.now() - a).total_seconds() * 1000
        print(f'\r time:{b}')
        return final_pred


if __name__ == '__main__':
    import pickle
    from torchvision.transforms import Normalize, ToTensor, Compose, RandomCrop
    # from models.SegFormer import SegFormer
    from models.UPerNet import UPerNet
    import cv2
    from utils import to_numpy, to_comb_image, un_normalise, check_module_prefix

    file = open('..\\ade20k_img.pkl', 'rb')
    img = pickle.load(file)
    file.close()

    path_to_chkpt = '..\\logging\\ADE20K\\20220326_185031_e1__upn_ConvNextT_sbn_DCms_cs_epochs127_bs16\\chkpts\\chkpt_epoch_126.pt'
    map_location = 'cuda:0'
    checkpoint = torch.load(str(path_to_chkpt), map_location)

    config = dict()

    config.update({'backbone': 'ConvNextT', 'out_stride': 32, 'pretrained': False, 'dataset':'ADE20K',
                   'pretrained_res':224, 'pretrained_dataset':'22k' , 'align_corners':False})

    model = UPerNet(config, 1)

    ret = model.load_state_dict(check_module_prefix(checkpoint['model_state_dict'], model), strict=False)
    print(ret)
    T = Compose([
                 ToTensor(),
                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 # RandomCrop(size=[512, 512])
                 ])

    with torch.no_grad():

        tta_model = TTAWrapperCTS(model, scale_list=[0.5], crop_size=(512, 512), strides=(341, 341), debug=True)  #  [0.75, 1.25, 1.5, 1.75, 2, 1.0]
        tta_model.cuda()
        tta_model.eval()
        x = T(img)
        # x  = x.cuda().float()
        y = tta_model.forward(x.unsqueeze(0).float())
        c = 1
