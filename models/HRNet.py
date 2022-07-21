"""
MIT License
Copyright (c) 2019 Microsoft
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models.utils import load_state_dict_from_url
from models.hrnet_config import MODEL_CONFIGS
from models.Projector import Projector
import numpy as np
from utils import DATASETS_INFO, printlog, get_rank
from utils import Logger as Log

# logger = logging.getLogger('hrnet_backbone')

__all__ = ['hrnet18', 'hrnet32', 'hrnet48', 'HRNet']


model_urls = {
    # all the checkpoints come from https://github.com/HRNet/HRNet-Image-Classification
    'hrnet18': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    'hrnet32': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
    'hrnet48': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None, align_corners=False):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.align_corners = align_corners

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            Log.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            Log.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            Log.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )

        layers = list()
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=self.align_corners)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self,
                 cfg,
                 norm_layer=None,
                 mixing_layer=False,
                 use_as_backbone=False,
                 return_all_scales=False,
                 align_corners=False,
                 dataset='CITYSCAPES',
                 experiment=1):
        super(HighResolutionNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        # if true then the final features maps from 4 resolutions
        # are fused into the highest one using a small CNN : see last_layer sequential below
        # todo clean up all input args to be passed here by config dict

        self.use_mxing_layer = mixing_layer
        self.dataset = dataset
        self.experiment = experiment
        self.use_as_backbone = use_as_backbone
        self.return_all_scales = return_all_scales
        self.align_corners = align_corners
        self.out_stride = 4
        self.projector_model = None  # todo
        if use_as_backbone and self.use_mxing_layer:
            self.num_classes = 0
        else:
            self.num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1]) - 1 \
                if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1].keys() \
                else len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1])

        # stem network
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = self.norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = self.norm_layer(64)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        # stage 2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # if self.use_mxing_layer and not self.use_as_backbone:
        #     # adds a multiscale mixing layer (bilinear up of 4 feature maps to 1/4 of input res and 1x1 conv)
        #     last_inp_channels = np.int(np.sum(pre_stage_channels))
        #     self.last_layer = nn.Sequential(
        #         nn.Conv2d(
        #             in_channels=last_inp_channels,
        #             out_channels=last_inp_channels,
        #             kernel_size=1,
        #             stride=1,
        #             padding=0),
        #         self.norm_layer(last_inp_channels),
        #         nn.ReLU(inplace=False),
        #         nn.Conv2d(
        #             in_channels=last_inp_channels,
        #             out_channels=self.num_classes,
        #             kernel_size=1,
        #             stride=1,
        #             padding=0)
        #     )

        # elif self.use_mxing_layer:
        #     # adds a multiscale mixing layer (bilinear up of 4 feature maps to 1/4 of input res and 1x1 conv)
        #     last_inp_channels = np.int(np.sum(pre_stage_channels) )
        #     self.last_layer = nn.Sequential(
        #         nn.Conv2d(
        #             in_channels=last_inp_channels,
        #             out_channels=last_inp_channels,
        #             kernel_size=1,
        #             stride=1,
        #             padding=0),
        #         self.norm_layer(last_inp_channels),
        #         nn.ReLU(inplace=False)
        #     )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        in_resolution = x.shape[-2:]  # (h,w)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # outputs = dict()
        # See note [TorchScript super()]
        # outputs['res2'] = x[0]  # 1/4
        # outputs['res3'] = x[1]  # 1/8
        # outputs['res4'] = x[2]  # 1/16
        # outputs['res5'] = x[3]  # 1/32
        assert self.use_as_backbone
        # if self.use_mxing_layer:
        # Upsampling and concatenating
        x0_size = x[0].shape[-2:]
        x1 = F.interpolate(x[1], size=x0_size, mode='bilinear', align_corners=self.align_corners)
        x2 = F.interpolate(x[2], size=x0_size, mode='bilinear', align_corners=self.align_corners)
        x3 = F.interpolate(x[3], size=x0_size, mode='bilinear', align_corners=self.align_corners)
        if self.return_all_scales:
            feats = [x[0], x[1], x[2], x[3]]
            # [s.shape for s in feats]
        x = torch.cat([x[0], x1, x2, x3], 1)

        if self.return_all_scales:
            return x, feats
        else:
            return x

class HRNet(nn.Module):
    eligible_backbones=['hrnet48']
    def __init__(self, config, experiment):
        super().__init__()
        self.config = config
        self.backbone_name = 'hrnet48'
        self.out_stride =  4
        self.dataset = config['dataset']
        self.norm = nn.BatchNorm2d
        assert(self.backbone_name in self.eligible_backbones), 'backbone must be in {}'.format(self.eligible_backbones)
        self.num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1]) - 1 if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1].keys() \
            else len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1])

        self.align_corners = config['align_corners'] if 'align_corners' in config  else True

        self.use_ms_projector = False
        self.projector_before_context = None
        self.backbone_cutoff = {'layer4': 'C5'}

        self.return_backbone_feats = False
        return_all_scales = True if 'ms_projector' in config else False

        # to return backbone features s4-s32 for tsne visualization
        if 'return_all_scales' in config:
            return_all_scales = config['return_all_scales']
            self.return_backbone_feats = True
            self.return_features = True

        self.backbone = hrnet48(self.config['pretrained'], mixing_layer=True, use_as_backbone=True,
                                return_all_scales=return_all_scales, align_corners=self.align_corners)

        self.backbone_out_channels = sum(self.backbone.stage4_cfg.NUM_CHANNELS)
        in_channels = self.backbone_out_channels  # 48 + 96 + 192 + 384

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            self.norm(in_channels),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        if 'projector' in config:
            self.return_features = True

            self.config['projector']['c_in'] = self.backbone_out_channels
            self.projector_model = Projector(config=self.config['projector'])
            printlog('added projector from {} to {}'.format(self.projector_model.c_in, self.projector_model.d))

        elif 'ms_projector' in config:
            self.return_features = True
            self.use_ms_projector = True
            self.ms_projector_scales = self.config['ms_projector']['scales'] if 'scales' in self.config['ms_projector'] else 4
            assert self.ms_projector_scales in [2,3,4], f'HRNet scales must be in [2,3,4] instead got {self.ms_projector_scales}'
            self.config['ms_projector']['c_in'] = self.backbone.stage4_cfg.NUM_CHANNELS[:self.ms_projector_scales]
            self.projector_model = Projector(config=self.config['ms_projector'])
            printlog(f'added {self.ms_projector_scales}'
                     f' projectors from {self.projector_model.c_in} to {self.projector_model.d}')

        else:
            self.projector_before_context = None
            self.projector_model = None
            self.return_features = False

        if 'return_all_scales' in config:
            self.return_features = True

    def forward(self, x):
        input_resolution = x.shape[-2:]  # input image resolution (H,W)
        # resnet path
        backbone_features = self.backbone.forward(x)
        # print(backbone_features.size())
        if self.use_ms_projector or self.return_backbone_feats:
            logits = self.cls_head(backbone_features[0])
        else:
            logits = self.cls_head(backbone_features)

        # print(logits.size())
        upsampled_logits = F.interpolate(logits, size=input_resolution, mode='bilinear', align_corners=self.align_corners)

        if self.projector_model:
            if self.use_ms_projector:
                proj_features = self.projector_model(backbone_features[1][:self.ms_projector_scales])
            else:
                proj_features = self.projector_model(backbone_features)

            if self.return_features:
                return upsampled_logits, proj_features
            else:
                return upsampled_logits
        else:
            if self.return_features:
                return upsampled_logits, backbone_features
            else:
                return upsampled_logits


def _hrnet(arch, pretrained, progress, **kwargs):
    model = HighResolutionNet(MODEL_CONFIGS[arch], **kwargs)
    if pretrained:
        if int(os.environ.get("mapillary_pretrain", 0)):
            printlog("load the mapillary pretrained hrnet-w48 weights.")
            model_url = model_urls['hrnet48_mapillary_pretrain']
        else:
            model_url = model_urls[arch]

        local_path_to_chkpt = None
        if os.path.isfile("/nfs/home/tpissas/data/pytorch_checkpoints/hrnet/hrnetv2_w48_imagenet_pretrained.pth"):
            local_path_to_chkpt = "/nfs/home/tpissas/data/pytorch_checkpoints/hrnet/hrnetv2_w48_imagenet_pretrained.pth"
        elif os.path.isfile("hrnetv2_w48_imagenet_pretrained.pth"):
            local_path_to_chkpt = "hrnetv2_w48_imagenet_pretrained.pth"

        if local_path_to_chkpt:
            state_dict = torch.load(local_path_to_chkpt)
        else:
            state_dict = load_state_dict_from_url(model_url,
                                                  progress=progress)
        ret = model.load_state_dict(state_dict, strict=False)

        model_dict = model.state_dict().keys()
        load_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        printlog(f'loading pretrained hrnet')
        printlog(f"checkpoint is at device {state_dict['conv1.weight'].device}, will be moved to {'cuda:%d' % get_rank()} (= process_rank)")
        printlog('Missing keys: {}'.format(list(set(model_dict) - set(load_dict))))
    return model


def hrnet18(pretrained=False, progress=True, **kwargs):
    r"""HRNet-18 model
    """
    return _hrnet('hrnet18', pretrained, progress,
                   **kwargs)


def hrnet32(pretrained=False, progress=True, **kwargs):
    r"""HRNet-32 model
    """
    return _hrnet('hrnet32', pretrained, progress,
                   **kwargs)


def hrnet48(pretrained=False, progress=True, **kwargs):
    r"""HRNet-48 model
    """
    return _hrnet('hrnet48', pretrained, progress,
                   **kwargs)


if __name__ == '__main__':
    Log.init()

    # device = torch.device('cuda')
    # torch.cuda.set_device(0)
    from torchvision.models._utils import IntermediateLayerGetter
    # net = hrnet48(True, mixing_layer=True, use_as_backbone=True)
    config = dict()
    config.update({'backbone': "hrnet48", 'pretrained': False,
                   'dataset':'CITYSCAPES', "align_corners":True, "return_all_scales":True})
    config.update({'out_stride': 4})
    config.update({"ms_projector": {"mlp": [[1, -1, 1], [1, 256, 1]], "d": 256, "use_bn": True, "scales":4,
                                    "before_context": False}})
    net = HRNet(config, 1)
    # layers = IntermediateLayerGetter(net, {'layer3': 'low', 'layer4': 'high'})
    x_ = torch.ones(size=(1, 3, 512, 512))
    f = net(x_)
    print(f)
    a = 1

    # 'hrnet48': {'FINAL_CONV_KERNEL': 1,
    #             'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'NUM_BLOCKS': [4], 'NUM_CHANNELS': [64],
    #                        'BLOCK': 'BOTTLENECK', 'FUSE_METHOD': 'SUM'},
    #             'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [48, 96],
    #                        'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'},
    #             'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [48, 96, 192],
    #                        'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'},
    #             'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'NUM_BLOCKS': [4, 4, 4, 4],
    #                        'NUM_CHANNELS': [48, 96, 192, 384], 'BLOCK': 'BASIC', 'FUSE_METHOD': 'SUM'}}
