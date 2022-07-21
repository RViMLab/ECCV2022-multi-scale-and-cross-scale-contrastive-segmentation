import torch
from torch import nn
import torch.nn.functional as f
from utils import DATASETS_INFO, conv3x3, printlog
from models.Projector import Projector
from models.Swin import SwinTransformer
from models.Swin import backbone_config as backbone_config_swin
from torchvision.models import resnet50, resnet101
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F
import json


class FPN(nn.Module):
    """FPN implementation, see https://github.com/CSAILVision/semantic-segmentation-pytorch"""
    def __init__(self, config, experiment):
        super().__init__()
        self.dropout = config['dropout_rate'] if 'dropout_rate' in config else 0.0
        self.align_corners = config['align_corners'] if 'align_corners' in config  else True
        self.dataset = config['dataset']
        self.num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1]) - 1 if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1].keys() \
            else len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1])
        self.pool_scales = [1, 2, 3, 6] if 'pool_scales' not in config else config['pool_scales']
        self.in_channels = config['input_channels']
        self.in_scales = config['input_scales']
        self.ppm_num_ch = 512 if 'ppm_num_ch' not in config else config['ppm_num_ch']
        self.fpn_num_ch = 512 if 'fpn_num_ch' not in config else config['fpn_num_ch']
        self.fpn_num_lvl = len(self.in_scales) if 'fpn_num_lvl' not in config else config['fpn_num_lvl']
        self.fpn_num_lvl = max(self.fpn_num_lvl, 1)  # Make sure no more lvls chosen than exist
        self.fpn_num_lvl = min(self.fpn_num_lvl, len(self.in_scales))  # Make sure no more lvls chosen than exist
        self.interpolate_result_up = True if 'interpolate_result_up' not in config else config['interpolate_result_up']
        self.return_features = True
        self.ppm_pooling = []
        self.ppm_conv = []
        for scale in self.pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(self.in_channels[-1], self.ppm_num_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.ppm_num_ch),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3(self.in_channels[-1] + len(self.pool_scales) * self.ppm_num_ch,
                                     self.fpn_num_ch, batch_norm=True, relu=True, stride=1)

        # FPN Module
        self.fpn_in = []
        for in_channel in self.in_channels[-self.fpn_num_lvl:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(in_channel, self.fpn_num_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.fpn_num_ch),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(self.fpn_num_lvl - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3(self.fpn_num_ch, self.fpn_num_ch, batch_norm=True, relu=True, stride=1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3(self.fpn_num_lvl * self.fpn_num_ch, self.fpn_num_ch, batch_norm=True, relu=True, stride=1),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(self.fpn_num_ch, self.num_classes, kernel_size=1)
        )

    def forward(self, conv_out):
        conv5 = conv_out[-1]  # [f.shape for f in conv_out]
        input_size = conv5.size()
        ppm_out = [conv5]
        for ppm_pool, ppm_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(ppm_conv(nn.functional.interpolate(
                ppm_pool(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        feature = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [feature]
        for i in range(2, self.fpn_num_lvl + 1):
            conv_x = conv_out[-i]
            conv_x = self.fpn_in[-i + 1](conv_x)  # lateral branch

            feature = nn.functional.interpolate(
                feature, size=conv_x.size()[2:], mode='bilinear', align_corners=self.align_corners)  # top-down branch
            feature = conv_x + feature

            fpn_feature_list.append(self.fpn_out[-i + 1](feature))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(2, self.fpn_num_lvl + 1):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[-i + 1],
                output_size,
                mode='bilinear', align_corners=self.align_corners))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        # if self.interpolate_result_up:
        #     x = f.interpolate(x, scale_factor=self.in_scales[-self.fpn_num_lvl], mode='bilinear', align_corners=self.align_corners)
        if self.return_features:
            return x, fpn_feature_list, fusion_out
        return x


class UPerNet(nn.Module):
    eligible_backbones = ['resnet50', 'resnet101', 'swinT', 'swinS', 'swinB', 'swinL', 'mobilenetv3',
                          'ConvNextT', 'ConvNextS', 'ConvNextB', 'ConvNextL']
    torchvision2paper_resnet_layer_name_mapping = {"layer1": "C2", "layer2": "C3", "layer3": "C4", "layer4": "C5"}
    valid_projector_positions = ['fpn', 'backbone', 'fused_feats']
    def __init__(self, config, experiment):
        super().__init__()
        self.config = config
        self.experiment = experiment
        self.out_stride =  32
        self.dataset = config['dataset']
        self.backbone_name = config['backbone']
        self.norm = nn.BatchNorm2d
        assert(self.backbone_name in self.eligible_backbones), 'backbone must be in {}'.format(self.eligible_backbones)
        self.num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1]) - 1 \
            if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1].keys() \
            else len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1])
        self.align_corners = config['align_corners'] if 'align_corners' in config  else True
        self.return_backbone_feats = False
        # projector + fpn settings (only resnet for now todo)
        self.backbone_cutoff = {"layer1": "C2", "layer2": "C3", "layer3": "C4", "layer4": "C5"}
        # self.backbone_cutoff = {"layer1": "C2", "layer2": "C3", "layer3": "C4", "layer4": "C5"}
        self.proj_feats = ['C2','C3','C4','C5']
        self.fpn_inputs = ['C2','C3','C4','C5']

        if 'return_all_scales' in config: # tsne related
            self.return_backbone_feats = config['return_all_scales']
            self.return_features = True

        self._get_backbone()
        self.fpn = FPN(config=self.config, experiment=self.experiment)
        self._get_aux_head()
        self._get_projector()

    def _get_aux_head(self):
        if 'aux_head' in self.config:
            self.aux_in_index = self.config['aux_head']['in_index']
            self.aux_in_channels = self.config['input_channels'][self.aux_in_index]
            self.aux_out_channels = self.config['aux_head'].get('out_channels', 256)
            self.aux_dropout = self.config['aux_head'].get('dropout_rate', 0.0)
            self.aux_head = nn.Sequential(
                nn.Conv2d(self.aux_in_channels, self.aux_out_channels, kernel_size=3, stride=1, padding=1),
                self.norm(self.aux_out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.aux_dropout),
                nn.Conv2d(self.aux_out_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
            self.get_intermediate = True
            printlog(f'adding auxiliary head at {self.aux_in_index}, with c_in {self.aux_in_channels}')
        else:
            self.in_index = None
            self.aux_head = None
            self.get_intermediate = False

    def _get_backbone(self):
        # backbone prep
        backbone_pretrained = True if 'pretrained' not in self.config else self.config['pretrained']
        if self.backbone_name == 'resnet50':
            self.backbone = IntermediateLayerGetter(resnet50(pretrained=backbone_pretrained), return_layers=self.backbone_cutoff)
            self.backbone_out_channels = self.backbone['layer4']._modules['2'].conv3.out_channels
            self.config['input_channels'] = [256, 512, 1024, 2048]
            self.config['input_scales'] = [4, 8, 16, 32]  # fpn scales
        elif self.backbone_name == 'resnet101':
            self.backbone = IntermediateLayerGetter(resnet101(pretrained=backbone_pretrained), return_layers=self.backbone_cutoff)
            self.backbone_out_channels = self.backbone['layer4']._modules['2'].conv3.out_channels
            self.config['input_channels'] = [256, 512, 1024, 2048]
            self.config['input_scales'] = [4, 8, 16, 32]  # fpn scales
        elif self.backbone_name in ['swinT', 'swinS', 'swinB', 'swinL']:
            # path_to_config = 'configs/backbones/swinTSB.json'
            backbone_settings = backbone_config_swin[self.backbone_name]
            backbone_settings['pretrained'] = backbone_pretrained
            self.backbone = SwinTransformer(**backbone_settings)
            self.config['input_channels'] = backbone_settings['out_channels']
            self.config['input_scales'] = [4, 8, 16, 32]  # fpn scales
        else:
            raise  NotImplementedError(f'{self.backbone_name}')

    def _get_projector(self):
        self.projector_position = None
        self.projector_model = None
        self.return_features = False
        self.use_ms_projector = False

        if 'projector' in self.config:
            self.return_features = True
            self.projector_position = 'fused_feats'
            self.config['projector']['c_in'] = self.backbone_out_channels
            self.projector_model = Projector(config=self.config['projector'])
            printlog(f'added projector from {self.projector_model.c_in} to {self.projector_model.d} | position {self.projector_position}')

        elif 'ms_projector' in self.config:

            self.ms_projector_scales = self.config['ms_projector']['scales'] if 'scales' in self.config['ms_projector'] else self.fpn.fpn_num_lvl

            self.return_features = True
            self.use_ms_projector = True
            self.projector_position = self.config['ms_projector']['position']
            if self.projector_position == 'backbone':
                self.config['ms_projector']['c_in'] = self.config['input_channels'][:self.ms_projector_scales]
            elif self.projector_position == 'fpn':
                self.config['ms_projector']['c_in'] = [self.fpn.fpn_num_ch]* self.ms_projector_scales
            self.projector_model = Projector(config=self.config['ms_projector'])
            printlog(f'added ms projectors from {self.projector_model.c_in} to {self.projector_model.d} | position {self.projector_position}')

    def forward(self, x):
        input_resolution = x.shape[-2:]  # input image resolution (H,W)
        backbone_features = self.backbone(x)
        if isinstance(backbone_features, dict):
            logits, fpn_feats, fpn_fused_feats = self.fpn([backbone_features[f] for f in self.fpn_inputs])
        else:
            logits, fpn_feats, fpn_fused_feats = self.fpn(backbone_features)

        kwargs_interp = {"size":input_resolution, "mode":"bilinear", "align_corners": self.align_corners}
        upsampled_logits = F.interpolate(logits, **kwargs_interp)

        if self.get_intermediate and self.aux_head:
            if isinstance(backbone_features, dict):
                proj_features = self.aux_head([backbone_features[f] for f in self.proj_feats][self.aux_in_index])
            interm_logits = self.aux_head(backbone_features[self.aux_in_index])
            upsampled_interm_logits = F.interpolate(interm_logits, **kwargs_interp)

        if self.projector_model:
            if self.use_ms_projector:
                # order high - low res
                if self.projector_position == 'backbone':
                    if isinstance(backbone_features, dict):
                        proj_features = self.projector_model([backbone_features[f] for f in self.proj_feats])
                    else:
                        proj_features = self.projector_model(backbone_features[:self.ms_projector_scales])

                elif self.projector_position == 'fpn':
                    proj_features = self.projector_model(fpn_feats)
            elif self.projector_position == 'fused_feats':
                proj_features = self.projector_model(fpn_fused_feats)

            if self.return_features:
                if self.get_intermediate:
                    return upsampled_interm_logits, upsampled_logits, proj_features
                else:
                    return upsampled_logits, proj_features
        else:
            if self.get_intermediate:
                if self.return_backbone_feats:
                    return upsampled_interm_logits, upsampled_logits, [backbone_features[f] for f in self.fpn_inputs]
                else:
                    return upsampled_interm_logits, upsampled_logits

            else:
                if self.return_backbone_feats:
                    return upsampled_logits, [backbone_features[f] for f in self.fpn_inputs][::-1]
                else:
                    return upsampled_logits


    def get_num_params(self):
        num_params = sum(p.numel() for p in self.enc_model.parameters() if p.requires_grad)
        if 'projector' in self.config:
            num_params += sum(p.numel() for p in self.projector_model.parameters() if p.requires_grad)
        num_params += sum(p.numel() for p in self.dec_model.parameters() if p.requires_grad)
        return num_params


if __name__ == '__main__':
    # device = torch.device('cuda')
    # torch.cuda.set_device(0)
    from torchvision.models._utils import IntermediateLayerGetter
    # net = hrnet48(True, mixing_layer=True, use_as_backbone=True)
    config = dict()
    # config.update({'backbone': "swinT", 'pretrained': True, 'dataset':'CITYSCAPES', "align_corners":True})
    # todo aux does not work with rensets
    # todo simplify forward to return staff as dict
    config.update({'backbone': "swinS", 'pretrained': False, 'pretrained_dataset': '22k', 'pretrained_res': '224',
                   'return_all_scales':True,
                   'dataset':'ADE20K', "align_corners":False})
    config.update({'out_stride': 32})
    config.update({'dropout_rate': 0.1})
    config.update({"aux_head___":{"in_index":3}, "dropout_rate":0.1})
    config.update({"ms_projector__": {"mlp": [[1, -1, 1]], "d": 256, "use_bn": True, "scales":4, "position":"fpn",
                                    "before_context": False}})
    net = UPerNet(config, 1)
    # layers = IntermediateLayerGetter(net, {'layer3': 'low', 'layer4': 'high'})
    # from torch.utils.tensorboard.writer import SummaryWriter
    # # train_writer = SummaryWriter()
    x_ = torch.ones(size=(1, 3, 256, 256))
    # # train_writer.add_graph(net, x_ .float())
    net.return_features = True
    with torch.no_grad():
        f = net(x_)

    print(f)
    a = 1


