import torch
from torch import nn
from torch.nn import functional as F
from utils import DATASETS_INFO, printlog
from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import os
from collections import OrderedDict
from models.Projector import Projector
from models.HRNet import hrnet48


class DeepLabv3(nn.Module):
    eligible_backbones = ['resnet18','resnet50', 'resnet101']
    torchvision2paper_resnet_layer_name_mapping = { "layer1":"C2", "layer2":"C3","layer3":"C4", "layer4":"C5"}
    layers_strides_resnet = { "layer1":0.5, "layer2":0.25,"layer3":0.125, "layer4":0.125}

    def __init__(self, config, experiment):
        super().__init__()
        self.config = config
        self.backbone_name = config['backbone'] if 'backbone' in config else 'resnet50'
        self.c_aspp = config['aspp']['channels'] if 'aspp' in config else 256
        self.out_stride = config['out_stride'] if 'out_stride' in config else 16
        self.dataset = config['dataset']
        self.align_corners = config['align_corners'] if 'align_corners' in config else True
        assert(self.out_stride in [8, 16, 32])
        if self.out_stride == 8:
            layer_2_stride, layer_3_stride, layer_4_stride = False, True, True
        elif self.out_stride == 16:
            layer_2_stride, layer_3_stride, layer_4_stride = False, False, True
        else:
            layer_2_stride, layer_3_stride, layer_4_stride = False, False, False
        striding = [layer_2_stride, layer_3_stride, layer_4_stride]
        assert(self.backbone_name in self.eligible_backbones), 'backbone must be in {}'.format(self.eligible_backbones)
        self.num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1]) - 1 \
            if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1].keys() \
            else len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1])
        # chop off fully connected layers from the backbone + load pretrained weights
        # we replace stride with dilation in resnet layer 4 to make output_stride = 8 (instead of higher by default)
        self.use_ms_projector = False
        self.backbone_cutoff = {'layer4': 'C5'}
        self.proj_feats = []
        if 'ms_projector' in config:
            if 'feats' in config['ms_projector']:
                feats = config['ms_projector']['feats']
                d = dict()
                for f in feats:
                    d[f] = self.torchvision2paper_resnet_layer_name_mapping[f]
                    self.proj_feats.append(d[f])
            else:
                # legacy version
                d = {'layer1': 'C2'}
                self.proj_feats.append('C2')
            self.backbone_cutoff.update(d)
            # self.backbone_cutoff.update({'layer1': 'C2', 'layer3': 'C4'})


        backbone_pretrained = True if 'pretrained' not in config else config['pretrained']
        if self.backbone_name == 'resnet50':
            self.backbone = IntermediateLayerGetter(resnet50(pretrained=backbone_pretrained,
                                                             replace_stride_with_dilation=striding),
                                                    return_layers=self.backbone_cutoff)
            self.backbone_out_channels = self.backbone['layer4']._modules['2'].conv3.out_channels

        elif self.backbone_name == 'resnet101':
            self.backbone = IntermediateLayerGetter(resnet101(pretrained=backbone_pretrained,
                                                              replace_stride_with_dilation=striding),
                                                    return_layers=self.backbone_cutoff)
            self.backbone_out_channels = self.backbone['layer4']._modules['2'].conv3.out_channels

        # define Atrous Spatial Pyramid Pooling layer
        mult = 2
        self.aspp = ASPP(c_in=self.backbone_out_channels, c_aspp=self.c_aspp, mult=mult, align_corners=self.align_corners)
        self.conv_out = nn.Conv2d(self.c_aspp, self.num_classes, kernel_size=1, stride=1)

        if 'projector' in config:
            self.return_features = True
            if config['projector']['before_context']:
                self.config['projector']['c_in'] = self.backbone_out_channels
            else:
                self.config['projector']['c_in'] = self.c_aspp
            self.projector_before_context = config['projector']['before_context']
            self.projector_model = Projector(config=self.config['projector'])
            printlog('added projector from {} to {}'.format(self.projector_model.c_in, self.projector_model.d))

        elif 'ms_projector' in config:
            self.return_features = True
            if self.backbone_name in ['resnet50','resnet101']:
                self.mid1_channels = 512 if 'layer2' in self.proj_feats else 256 # layer1 = 256 else layer2 = 512

                #self.backbone['layer1']._modules['2'].conv3.out_channels
                # self.mid1_channels = 256 #self.backbone['layer1']._modules['2'].conv3.out_channels
                self.mid2_channels = 1024 #self.backbone['layer3']._modules['5'].conv3.out_channels
            else:
                raise NotImplementedError(f'{self.backbone_name}')

            if len(self.proj_feats)==2:
                self.config['ms_projector']['c_in'] = [self.mid1_channels, self.backbone_out_channels]
            elif len(self.proj_feats)==3:
                self.config['ms_projector']['c_in'] = [self.mid1_channels, self.mid2_channels, self.backbone_out_channels]
            else:
                raise NotImplementedError(f'invalid : {self.proj_feats}')

            self.projector_model = Projector(config=self.config['ms_projector'])
            printlog('added ms projectors from {} to {}'.format(self.projector_model.c_in, self.projector_model.d))
            self.use_ms_projector = True
            self.projector_before_context = True  # this projector can only be applied before context

        else:
            self.projector_before_context = None
            self.projector_model = None
            self.return_features = False

    def forward(self, x):
        input_resolution = x.shape[-2:]  # input image resolution (H,W)
        backbone_features = self.backbone.forward(x)
        aspp_features = self.aspp.forward(backbone_features['C5'])
        logits = self.conv_out(aspp_features)
        upsampled_logits = F.interpolate(logits, size=input_resolution, mode='bilinear', align_corners=True)

        if self.projector_model:
            if self.projector_before_context:
                if self.use_ms_projector:
                    # [f.shape for f in [backbone_features[f] for f in self.proj_feats]]
                    proj_features = self.projector_model([backbone_features[f] for f in self.proj_feats])
                    # proj_features = self.projector_model([backbone_features['C3'], backbone_features['C5']])
                else:
                    proj_features = self.projector_model(backbone_features['C5'])
            else:
                proj_features = self.projector_model(aspp_features)

            # [f.shape for f in proj_features]

            if self.return_features:
                return upsampled_logits, proj_features
            else:
                return upsampled_logits
        else:
            return upsampled_logits

    def print_params(self):
        # just for debugging
        for w in self.state_dict():
            print(w, "\t", self.state_dict()[w].size())


class ASPP(nn.Module):
    # Atrous Spatial Pyramid Pooling

    def __init__(self, c_in, c_aspp, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1, align_corners=True):
        super(ASPP, self).__init__()
        self._c_in = c_in
        self._c_aspp = c_aspp
        self.align_corners=align_corners

        # image level features
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.aspp1 = conv(c_in, c_aspp, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(c_in, c_aspp, kernel_size=3, stride=1, dilation=int(6*mult), padding=int(6*mult), bias=False)
        self.aspp3 = conv(c_in, c_aspp, kernel_size=3, stride=1, dilation=int(12*mult), padding=int(12*mult), bias=False)
        self.aspp4 = conv(c_in, c_aspp, kernel_size=3, stride=1, dilation=int(18*mult), padding=int(18*mult), bias=False)
        self.aspp5 = conv(c_in, c_aspp, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(c_aspp, momentum)
        self.aspp2_bn = norm(c_aspp, momentum)
        self.aspp3_bn = norm(c_aspp, momentum)
        self.aspp4_bn = norm(c_aspp, momentum)
        self.aspp5_bn = norm(c_aspp, momentum)
        self.conv2 = conv(c_aspp * 5, c_aspp, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm(c_aspp, momentum)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear', align_corners=self.align_corners)(x5)
        x = torch.cat((x1, x2, x3, x4, x5), 1)  # concatenate along the channel dimension
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    config = dict()
    config.update({'backbone': 'resnet101', 'pretrained': True, 'dataset':'ADE20K'})
    config.update({'out_stride': 8})
    config.update({"ms_projector": {"mlp": [[1, -1, 1]],
                                    "feats":["layer1", "layer3", "layer4" ],
                                    "d": 256,
                                    "use_bn": True,
                                    "before_context": True}})
    import pathlib
    # checkpoint = torch.load("C:\\Users\\Theodoros Pissas\\PycharmProjects\\pytorch_checkpoints\\ReSim\\resim_c4_backbone_200ep.pth.tar")
    a = torch.ones(size=(1, 3, 256, 256))

    model = DeepLabv3(config, 1)
    # model.print_params()
    model.eval()
    b = model.forward(a)
    a = 1

