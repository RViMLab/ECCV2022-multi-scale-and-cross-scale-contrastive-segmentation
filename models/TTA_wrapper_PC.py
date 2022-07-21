import torch
from torch.nn.parallel import DistributedDataParallel as ddp
from torch import nn
from torch.nn import functional as F
import datetime
import cv2
import numpy as np
from utils import printlog, to_numpy, to_comb_image
from models import TTAWrapper


class TTAWrapperPC(TTAWrapper):
    def __init__(self, model, scale_list):
        super().__init__(model, scale_list)
        self.num_classes = 59
        self.crop_size = [512,512]
        self.base_size = 520

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

        # to_comb_image(un_normalise(image[0]), torch.argmax(pred[0], 0), None, 1, 'PASCALC', save=f'pred_{self.ind}_scale_{scale}_{id_}.png')
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

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def forward(self, x, ind=0):
        self.ind = ind
        a = datetime.datetime.now()
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.tensor), f'x input must be a tensor instead got {type(x)}'
        batch, _, ori_height, ori_width = x.size()
        assert batch == 1, "only supporting batchsize 1."
        # x is BCHW
        image = to_numpy(x)[0].transpose((1, 2, 0)).copy(       )
        # x is HWC
        stride_h = int(self.crop_size[0] * 2.0/3.0)
        stride_w = int(self.crop_size[1] * 2.0/3.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        padvalue = -1.0 * np.array(mean)/np.array(std)

        for scale in self.scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            # cv2.imshow(f'scale {scale}', new_img)
            height, width = new_img.shape[:-1]

            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width,
                                         self.crop_size, padvalue)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(new_img.cuda(), flip=True, scale=scale, id_=1)
                preds = preds[:, :, 0:height, 0:width]
            else:

                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width,
                                             self.crop_size, padvalue)

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
                        # h0 = max(int(h1 - self.crop_size[0]), 0)
                        # w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]

                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img,
                                                      h1-h0,
                                                      w1-w0,
                                                      self.crop_size,
                                                      padvalue)

                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(crop_img.cuda(), flip=True, scale=scale, id_= id_)
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

        # final_pred = F.interpolate(
        #     final_pred, ori_shape,
        #     mode='bilinear', align_corners=self.model.align_corners
        # )

        # to_comb_image(un_normalise(x[0]), torch.argmax(final_pred[0], 0), None, 1, 'PASCALC', save=f'final_{self.ind}.png')
        b = (datetime.datetime.now() - a).total_seconds() * 1000
        print(f'\r time:{b}')
        return final_pred


if __name__ == '__main__':
    import pickle
    from torchvision.transforms import Normalize, ToTensor, Compose, RandomCrop
    from models.HRNet import HRNet
    import cv2
    from utils import to_numpy, to_comb_image, un_normalise, check_module_prefix
    from datasets import PascalC
    data_path = r'C:\Users\Theodoros Pissas\Documents\tresorit\PASCALC/'
    from torchvision.transforms import ToTensor
    import PIL.Image as Image
    d = {"dataset":'PASCALC', "experiment":1}

    # file = open('..\\img_cts.pkl', 'rb')
    # img = pickle.load(file)
    # file.close()

    from utils import parse_transform_lists
    import json
    path_to_config = '../configs/dlv3_contrastive_PC.json'
    with open(path_to_config, 'r') as f:
        config = json.load(f)

    transforms_list_val = config['data']['transforms_val']
    # if 'torchvision_normalise' in transforms_list_val:
    #     del transforms_list_val[-1]
    transforms_values_val = config['data']['transform_values_val']
    transforms_dict_val = parse_transform_lists(transforms_list_val, transforms_values_val, dataset='PASCALC', experiment=1)
    valid_set = PascalC(root=data_path,
                        debug=False,
                        split='val',
                        transforms_dict=transforms_dict_val)

    issues = []
    valid_set.return_filename = True

    # if i ==5:
    #     break
    path_to_chkpt = '..\\logging/PASCALC/20211216_072315_e1__hrn_200epochs_hr48_sbn_DCms_cs/chkpts/chkpt_epoch_199.pt'
    # path_to_chkpt = '..\\logging/PASCALC/20211215_213857_e1__hrn_200epochs_hr48_sbn_CE/chkpts/chkpt_epoch_199.pt'
    map_location = 'cuda:0'
    checkpoint = torch.load(str(path_to_chkpt), map_location)
    torch.manual_seed(0)
    config = dict()
    config.update({'backbone': 'hrnet48', 'out_stride': 4, 'pretrained': True, 'dataset':'PASCALC'})
    model = HRNet(config, 1)
    msg = model.load_state_dict(check_module_prefix(checkpoint['model_state_dict'], model), strict=False)
    print(msg)

    for i, ret in enumerate(valid_set):
        img = ret[0]
        ori_shape = ret[2]['original_labels'].shape[-2:]
        with torch.no_grad():
            tta_model = TTAWrapperPC(model, scale_list=[0.75, 0.5, 1.5])
            tta_model.cuda()
            tta_model.eval()
            x = img
            y = tta_model.forward(x.unsqueeze(0).float(), ori_shape, i)
            c = 1