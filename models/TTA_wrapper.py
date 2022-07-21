import torch
from torch.nn.parallel import DistributedDataParallel as ddp
from torch import nn
from torch.nn import functional as F
import datetime
import cv2
from utils import printlog


class TTAWrapper(nn.Module):
    """
    hard-coding common scaling and flipping protocol for simplicity
    """
    def __init__(self, model, scale_list=None, flip=True):
        super().__init__()
        self.scales = scale_list# 1.5, 1.75, 2] # 1.5, 1.75, 2]
        self.flip = flip
        if 1.0 not in self.scales:
            self.scales.append(1.0)
        if isinstance(model, ddp):
            self.model = ddp.module
        else:
            self.model = model

        self.align_corners = self.model.align_corners if hasattr(self.model, 'align_corners') else True

        printlog(f'*** TTA wrapper with flip : [{flip}] --- scales : {self.scales} -- align_corners:{self.align_corners}')

    def maybe_resize(self, x, scale, in_shape):
        """

        :param x: B,C,H,W
        :param scale: if s in R+ resizes the image to s*in_shape,
                      if s=1 then return x,
                      if s=-1 then resize image to in_shape
        :param in_shape:
        :return:
        """
        scaled_shape = [int(scale * in_shape[0]), int(scale * in_shape[1])]
        if scale != 1.0 and scale > 0:
            x = F.interpolate(x, size=scaled_shape, mode='bilinear', align_corners=self.align_corners)
        elif scale == -1:
            x = F.interpolate(x, size=in_shape, mode='bilinear', align_corners=self.align_corners)
        else:
            x = x.clone()
        return x

    def maybe_flip(self, x, f):
        if f == 0:
            x_f = torch.flip(x, dims=[3]) # clones
        else:
            x_f = x.clone()
        return x_f

    def forward(self, x, **kwargs):
        if isinstance(x, tuple):
            x = x[0]
            assert isinstance(x, torch.tensor), f'x input must be a tensor instead got {type(x)}'

        a = datetime.datetime.now()
        assert len(x.shape)==4, 'input must be B,C,H,W'
        flag_first = True # flag for the first iteration of the nested loop]
        in_shape=x.shape[2:4]
        out_shape = [1, self.model.num_classes] + list(in_shape)
        y_merged = torch.zeros(size=out_shape).cuda()
        for f in range(2):
            x_f = self.maybe_flip(x, f) # flip
            for s in self.scales:
                x_f_s = self.maybe_resize(x_f, s, in_shape) # resize
                y = self.model(x_f_s) # forward
                y = self.maybe_flip(y, f) # unflip
                y_merged += self.maybe_resize(y, -1, in_shape) # un-resize

        b = (datetime.datetime.now() - a).total_seconds() * 1000
        # print('time taken for tta {:.5f}'.format(b))
        y_merged =  y_merged/(2*len(self.scales))
        # cv2.imshow('final', to_comb_image(un_normalise(x[0]), torch.argmax(y_merged[0], 0), None, 1, 'CITYSCAPES'))
        return y_merged





if __name__ == '__main__':
    import pickle
    from torchvision.transforms import Normalize, ToTensor, Compose, RandomCrop
    from models.HRNet import HRNet
    from models.UPerNet import UPerNet
    import cv2
    from utils import to_numpy, to_comb_image, un_normalise, check_module_prefix

    file = open('..\\img_cts.pkl', 'rb')
    img = pickle.load(file)
    file.close()

    path_to_chkpt = '..\\logging\\ADE20K\\20220326_185031_e1__upn_ConvNextT_sbn_DCms_cs_epochs127_bs16\\chkpts\\chkpt_epoch_126.pt'
    map_location = 'cuda:0'
    checkpoint = torch.load(str(path_to_chkpt), map_location)

    config = dict()
    config.update({'backbone': 'ConvNextT', 'out_stride': 32, 'pretrained': True, 'dataset':'ADE20K'})
    model = UPerNet(config, 1)

    ret = model.load_state_dict(check_module_prefix(checkpoint['model_state_dict'], model), strict=False)
    print(ret)
    T = Compose([
                 ToTensor(),
                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 RandomCrop(size=[512, 512])
                 ])

    with torch.no_grad():
        tta_model = TTAWrapper(model, scale_list=[0.5, 1.5])
        tta_model.cuda()
        tta_model.eval()
        x = T(img)
        x  = x.cuda().float()
        y = tta_model.forward(x.unsqueeze(0))
        c = 1

