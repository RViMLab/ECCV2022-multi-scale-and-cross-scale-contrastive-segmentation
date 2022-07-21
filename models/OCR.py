import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Projector import Projector
from torchvision.models import resnet101, resnet50, resnet18, resnet34
from models.HRNet import hrnet48
from torchvision.models._utils import IntermediateLayerGetter
from utils import DATASETS_INFO, is_distributed, printlog


class OCRNet(nn.Module):
    eligible_backbones = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'hrnet48']
    # reference to change in naming between torchvision and He et al resnet paper figure (see misc/figs/ResNetSpecs.PNG)
    torchvision2paper_resnet_layer_name_mapping = { "layer1":"C2", "layer2":"C3","layer3":"C4", "layer4":"C5"}
    layers_strides_resnet = { "layer1":0.5, "layer2":0.25,"layer3":0.125, "layer4":0.125}
    layers_strides_hrnet = { "layer1":1/4, "layer2":1/8,"layer3":1/16, "layer4":1/32}

    # Illustration of OCRNet architecture
    #                                               [General]
    # Backbone - layer1 - layer2 - (layer3) - (layer4) -- conv-bn-relu -- conv-bn-relu --->--{
    #                                  |                                                      [OCR]-> conv(cls)
    #                                  L -- conv-bn-relu -- conv(cls) --------->(m)------->--{              |
    #                                                                            |                       upsample
    #                                                                         upsample                      |
    #                                                                            |                         loss
    #                                                                          loss
    #
    #                                                 [OCR]
    #
    #                                                                Q,K,V Transform
    #      x -->                                     (B,C,H,W) x ---> {Q}   SpatialOCR -- x_object_contextual (B,C,H,W)
    #            {Spatial Gather} - global object representation ---> {K,V}
    #      m -->

    def __init__(self, config, experiment):
        super().__init__()
        self.config = config
        self.dataset = config['dataset']
        self.backbone_name = config['backbone'] if 'backbone' in config else 'resnet50'
        self.out_stride = config['out_stride'] if 'out_stride' in config else 8
        self.norm = config['norm'] if 'norm' in config else nn.BatchNorm2d
        assert(self.backbone_name in self.eligible_backbones), 'backbone must be in {}'.format(self.eligible_backbones)
        self.num_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1]) - 1 if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1].keys() \
            else len(DATASETS_INFO[self.dataset].CLASS_INFO[experiment][1])

        self.align_corners = config['align_corners'] if 'align_corners' in config  else True
        self.relu = nn.ReLU(inplace=True)
        self.dropout = config['dropout'] if 'dropout' in config else 0.0
        # if true,  forward() returns up intermediate logits, up final logits, else only up final logits are returned
        self.backbone_pretrained = True if 'pretrained' not in config else config['pretrained']
        self.get_intermediate = True

        self.return_all_scales = True if 'ms_projector' in config else False

        self._get_backbone()
        self._get_ocr()
        self._get_proj()


    def _get_backbone(self):
        self.backbone_cutoff = None
        if 'resnet' in self.backbone_name:
            # resnet_group = 1 if 'resnet50' in self.backbone_name or 'resnet101' in self.backbone_name else 0
            assert(self.out_stride in [8, 16, 32])
            if self.out_stride == 8:
                layer_2_stride, layer_3_stride, layer_4_stride = False, True, True
            elif self.out_stride == 16:
                layer_2_stride, layer_3_stride, layer_4_stride = False, False, True
            else:
                layer_2_stride, layer_3_stride, layer_4_stride = False, False, False
            strides = [layer_2_stride, layer_3_stride, layer_4_stride]

            self.backbone_cutoff = {'layer3': 'C4', 'layer4': 'C5'}
            if 'ms_projector' in self.config:
                self.backbone_cutoff.update({'layer1':'C2'})
                #

            resnet_class = globals()[self.backbone_name]
            self.backbone = IntermediateLayerGetter(resnet_class(pretrained=self.backbone_pretrained,
                                                                 replace_stride_with_dilation=strides),
                                                    return_layers=self.backbone_cutoff)
            if self.backbone_name=='resnet50':
                self.high_level_channels = self.backbone['layer4']._modules['2'].conv3.out_channels
                self.low_level_channels = self.backbone['layer3']._modules['2'].conv3.out_channels
            elif self.backbone_name=='resnet101':
                # case of resnet 18 or 34: blocks have 2 modules and 2 convs in each
                self.high_level_channels = self.backbone['layer4']._modules['2'].conv3.out_channels
                self.low_level_channels = self.backbone['layer3']._modules['22'].conv3.out_channels

        elif 'hrnet' in self.backbone_name:
            self.backbone_cutoff = None
            self.backbone = hrnet48(pretrained=self.backbone_pretrained, mixing_layer=True, use_as_backbone=True,
                                    return_all_scales=self.return_all_scales, align_corners=self.align_corners)
            # self.backbone_out_channels = sum(self.backbone.stage4_cfg.NUM_CHANNELS)
            self.high_level_channels = sum(self.backbone.stage4_cfg.NUM_CHANNELS) # 720
            self.low_level_channels = None
        else:
            raise NotImplementedError(f'{self.backbone_name} not implemented')

    def _get_ocr(self):
        # maps backbone final features to 512 channels
        self.ocr_dim = 512
        self.conv_high_map = nn.Sequential(
            nn.Conv2d(self.high_level_channels, 512, kernel_size=3, stride=1, padding=1),
            self.norm(512),
            self.relu
        )

        self.interm_pred_c_in = self.low_level_channels if 'resnet' in self.backbone_name else self.high_level_channels

        # maps layer3 features to intermediate logits
        self.interm_prediction_head = nn.Sequential(
            nn.Conv2d(self.interm_pred_c_in, 512, kernel_size=3, stride=1, padding=1),
            self.norm(512),
            self.relu,
            nn.Dropout2d(self.dropout),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        # ocr
        self.spatial_gather = SpatialGatherModule(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512, key_channels=256,
                                                  out_channels=512, scale=1,
                                                  dropout=self.dropout, norm=self.norm,
                                                  align_corners=self.align_corners)
        # output
        self.conv_out = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, bias=True)

    def _get_proj(self):
        # Make projector, if applicable
        self.projector_input_feats = []

        if 'projector' in self.config:
            self.return_features = True
            self.use_ms_projector = False
            self.projector_before_context = self.config['projector']['before_context']

            if self.projector_before_context:
                self.config['projector']['c_in'] =  self.high_level_channels
            else:
                self.config['projector']['c_in'] = self.ocr_dim
            self.projector_model = Projector(config=self.config['projector'])
            printlog('added projector from {} to {}'.format(self.projector_model.c_in, self.projector_model.d))

            self.projector_input_feats = 'C5' if self.projector_before_context else 'ocr'

        elif 'ms_projector' in self.config:
            self.return_features = True
            self.use_ms_projector = True
            self.projector_before_context = True  # ms projector can only be applied before context

            if self.backbone_name == 'resnet50':
                self.mid1_channels = self.backbone['layer1']._modules['2'].conv3.out_channels
                self.mid2_channels = self.backbone['layer2']._modules['3'].conv3.out_channels

            elif self.backbone_name == 'resnet101':
                self.mid1_channels = self.backbone['layer1']._modules['2'].conv3.out_channels
                self.mid2_channels = self.backbone['layer2']._modules['3'].conv3.out_channels


            if 'resnet' in self.backbone_name:
                # todo adds projector at layer1 and layer4
                #  if self.config['ms_projector']['scales'] == 3:
                #     self.config['ms_projector']['c_in'] = [self.mid1_channels, self.mid2_channels, self.high_level_channels]
                #  else:
                self.config['ms_projector']['c_in'] = [self.mid1_channels, self.high_level_channels]
                self.projector_input_feats = ['C2', 'C5']

            elif 'hrnet' in self.backbone_name:
                self.ms_projector_scales = 4 # todo un-hardcode this
                printlog(f'using default number of scales {self.ms_projector_scales} for ms_projector of hrnet')
                self.config['ms_projector']['c_in'] = self.backbone.stage4_cfg.NUM_CHANNELS[:self.ms_projector_scales]
                self.projector_model = Projector(config=self.config['ms_projector'])
                printlog(f'added {self.ms_projector_scales}'
                         f' projectors from {self.projector_model.c_in} to {self.projector_model.d}')
                self.projector_input_feats = ['stride4', 'stride8', 'stride16', 'stride32']

            self.projector_model = Projector(config=self.config['ms_projector'])
            printlog('added ms projectors from {} to {}'.format(self.projector_model.c_in, self.projector_model.d))

        else:
            self.use_ms_projector = False
            self.projector_before_context = None
            self.projector_model = None
            self.return_features = False


    def forward(self, x):
        input_resolution = x.shape[-2:]  # input image resolution (H,W)
        backbone_features = self.backbone(x)
        # print(f'high_features {backbone_features['C5'].size()}')
        # print(f'low_features  {backbone_features['C5'].size()}')
        if 'resnet' in self.backbone_name:
            intermediate_logits = self.interm_prediction_head(backbone_features['C4'])
            # print('intermediate_logits {}'.format(intermediate_logits.size()))
            x_high = self.conv_high_map(backbone_features['C5'])
        else:
            if isinstance(backbone_features, list) or isinstance(backbone_features, tuple):
                intermediate_logits = self.interm_prediction_head(backbone_features[0])
                x_high = self.conv_high_map(backbone_features[0])
            else:
                intermediate_logits = self.interm_prediction_head(backbone_features)
                x_high = self.conv_high_map(backbone_features)

        object_global_representation = self.spatial_gather(x_high, intermediate_logits)
        # print(f'object_global_representation {object_global_representation.size()}')

        ocr_representation = self.spatial_ocr_head(x_high, object_global_representation)
        # print(f'ocr_representation {ocr_representation.size()}')

        logits = self.conv_out(ocr_representation)
        # print(f'logits {logits.size()}')

        kwargs_interp = {"size":input_resolution, "mode":"bilinear", "align_corners": self.align_corners}

        up_logits = F.interpolate(logits, **kwargs_interp)

        outputs = [] #  (!) order must be (!) [interm_up_logits (optional), up_logits, proj_feats (optional)] (!)
        if self.get_intermediate:
            interm_up_logits = F.interpolate(intermediate_logits, **kwargs_interp)
            outputs.append(interm_up_logits)

        outputs.append(up_logits)

        if self.projector_model:
            if self.projector_before_context:
                if 'hrnet' in self.backbone_name:
                    feats = backbone_features[1][:self.ms_projector_scales]
                    # [f.shape for f in feats]
                elif 'resnet' in self.backbone_name:
                    feats = [backbone_features[f] for f in backbone_features if f in self.projector_input_feats]
                    # [f.shape for f in feats]
                proj_features = self.projector_model(feats)
            else:
                proj_features = self.projector_model(ocr_representation)

            if self.return_features:
                outputs.append(proj_features)

        if not self.get_intermediate and not self.return_features:
            assert len(outputs) == 1, f'when get_intermediate and' \
                                      f' return_features are False outputs must be of len 1 instead got {len(outputs)}'
            return outputs[0]
        else:
            return outputs


    def print_params(self):
        # just for debugging
        for w in self.state_dict():
            print(w, "\t", self.state_dict()[w].size())


class SpatialGatherModule(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGatherModule, self).__init__()
        self.cls_num = cls_num  # K:=cls_num
        self.scale = scale

    def forward(self, feats, probs):
        # probs: B, K, H, W  and feats: B, C, H, W
        batch_size = probs.size(0)

        probs = probs.view(batch_size, self.cls_num, -1)  # B, K, N , N:=H*W
        feats = feats.view(batch_size, feats.size(1), -1)  # B, C, N
        feats = feats.permute(0, 2, 1)  # B, N, C
        probs = F.softmax(self.scale * probs, dim=2)  # B, K, N

        ocr_context = torch.matmul(probs, feats) \
            .permute(0, 2, 1).unsqueeze(3)
        # B, K, N * B, N, C = B, K, C then B, C, K, 1
        return ocr_context


class ObjectAttentionBlock2D(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 norm=nn.BatchNorm2d,
                 aling_corners=True):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.relu = nn.ReLU(inplace=True)
        self.norm = norm

        # φ
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu,
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu
        )

        # ψ
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu,
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu
        )

        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.key_channels),
            self.relu
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm(self.in_channels),
            self.relu
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # conv of the backbone features to map to key_channels
        # in_channels := C
        # key_channels := C_key
        # x: B, C, H, W
        # proxy: B, C, C_cls, 1 # SpatialGather(x, prediction)

        # φ()
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        # x : B, C, H, W
        # query : B, C_key, H * W | N := H*W
        query = query.permute(0, 2, 1)
        # query : B, N, C_key

        # ψ()
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        # proxy: B, C, C_cls, 1
        # key: B, C_key, C_cls

        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        # proxy: B, C, C_cls, 1
        # value: B, C_key, C_cls
        value = value.permute(0, 2, 1)
        # value: B, C_cls, C_key

        sim_map = torch.matmul(query, key)
        # B, N, C_key * B, C_key, C_cls = B, N, C_cls
        # norm, softmax
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)  # along the C_cls dim
        # sim_map = B, N, C_cls

        # add bg context ...
        context = torch.matmul(sim_map, value)
        # context =  B, N, C_cls * B, C_cls, C_key = B, N, C_key
        context = context.permute(0, 2, 1).contiguous()
        # context = B, C_key, N
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        # context = B, C_key, H, W
        context = self.f_up(context)
        # context = B, C, H, W
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=self.align_corners)
        return context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.0,
                 norm=nn.BatchNorm2d,
                 align_corners=True):
        super(SpatialOCR_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           norm,
                                                           align_corners)
        _in_channels = 2 * in_channels
        # augmented representation: concatenate feat and ocr_feats and conv(1x1,512)
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            norm(out_channels),
            self.relu,
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


if __name__ == '__main__':
    c = dict()
    c.update({'backbone': 'resnet50'})
    c.update({'out_stride': 8,
              'align_corners':True,
              'dataset': 'CITYSCAPES',
              'projector___': {'mlp': [[1, -1, 1], [1, 256, 1]], 'd': 256, 'before_context': True},
              "ms_projector": {"mlp": [[1, -1, 1]], "scales": 4, "d": 256, "use_bn": True, "before_context": True}
              })
    # -c configs/HRNet_contrastive_CTS.json  -u theo -cdn False  -w 0 -so -m inference
    a = torch.ones(size=(1, 3, 128, 128))
    model = OCRNet(c, 1)
    # model.print_params()
    model.eval()
    interm, end, proj_feats = model.forward(a)
    print('interm:', interm.shape)
    print('end:', end.shape)

    if isinstance(proj_feats, list):
        print([f.shape for f in proj_feats])
    else:
        print('proj_feats:', proj_feats.shape)

