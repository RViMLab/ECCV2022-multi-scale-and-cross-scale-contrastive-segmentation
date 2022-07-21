import torch
from torch import nn
from typing import Union
from models.Transformers import SelfAttention
from utils import printlog

class Projector(nn.Module):

    def __init__(self, config):
        """ module that maps encoder features to a d-dimensional space
            if can be a single conv-linear (optionally) preceded by an fcn with conv-relu layers
         """
        super().__init__()
        self.d = config['d'] if 'd' in config else 128  # projection dim
        self.c_in = config['c_in']  # input features channels (usually == output channels of resnet backbone)
        assert isinstance(self.c_in, list) or isinstance(self.c_in, int)
        # config['mlp'] list of [k,c] for Conv-Relu layers, if empty only applies Conv(c_in, d, k=1)
        self.mlp = config['mlp'] if 'mlp' in config else []
        self.use_bn = config['use_bn'] if 'use_bn' in config else False
        self.transformer = config['trans'] if 'trans' in config else False
        self.heads = config['heads'] if 'heads' in config else 1

        if isinstance(self.c_in, list):
            self.is_ms = True
            self._create_ms_mlp()
        else:
            self.is_ms = False # whether the projector is multiscale
            self._create_mlp(self.c_in)

    def _create_ms_mlp(self):
        printlog('** creating ms projector **')
        for feat_id, c_in in enumerate(self.c_in):
            printlog(f'* scale {feat_id} feats: {c_in}')
            self._create_mlp(c_in, feat_id)

    def _create_mlp(self, c_in:int, feat_id:Union[list,int]=''):
        # sanity checks
        assert(isinstance(self.mlp, list)), 'config["mlp"] must be [[k_1, c_1, s_1], ., [k_n, c_n, s_n]] or [] ' \
                                            'k_i is kernel (k_i x k_i) c_i is channels and s_i is stride'
        first_layer_has_cout_equal_to_cin = False
        if len(self.mlp):
            for layer in self.mlp:
                assert(isinstance(layer, list)), f'elements of layer definition list must be lists instead got {layer}'
                assert(len(layer) == 3 and layer[2] in [1, 2]), 'must provide list of lists of 3 elements each' \
                                                                '[kernel, channels, stride] instead {}'.format(layer[2])
                if layer[1]>0:
                    assert(layer[0] < layer[1]), 'kernel size is first element of list, got {} {}'.format(layer[0], layer[1])

        self.convs = []
        c_prev = c_in
        if len(self.mlp):
            for layer_id, (k, c_out, s) in enumerate(self.mlp):
                if layer_id == 0 and c_out==-1:
                    c_out = c_prev
                printlog('Projector creating conv layer, k_{}/c_{}/s_{}'.format(k, c_out, s))
                # if use_bn --> do not use bias
                # p = (k + (k - 1) * (d - 1) - s + 1) // 2
                p = (k - s + 1) // 2
                self.convs.append(nn.Conv2d(c_prev, c_out, kernel_size=k, stride=s,
                                            padding=p, bias=not self.use_bn))
                self.convs.append(nn.ReLU(inplace=True))
                if self.use_bn:
                    self.convs.append(nn.BatchNorm2d(c_out, momentum=0.0003))
                c_prev = c_out
        if self.transformer:
            sa = SelfAttention(dim = c_prev, heads=self.heads)
            printlog('Projector creating transformer layer, heads_{}/c_{}'.format(self.heads, c_prev))
            self.convs.append(sa)

        printlog('Projector creating linear layer, k_{}/c_{}/s_{}'.format(1, self.d, 1))
        self.convs.append(nn.Conv2d(c_prev, self.d, kernel_size=1, stride=1))
        setattr(self, f'project{feat_id}', nn.Sequential(*self.convs))

    def forward(self, x:Union[list, torch.tensor]):
        # # x are features of shape NCHW
        # x = x / torch.norm(x, dim=1, keepdim=True)  # Normalise along C: features vectors now lie on unit hypersphere
        if self.is_ms:
            outs = []
            assert(isinstance(x, list) or isinstance(x, tuple)), f'if multiscale projector is used a list is expected as input instead got {type(x)}'
            for feat_id, x_i in enumerate(x):
                x_i = getattr(self, f'project{feat_id}')(x_i)
                outs.append(x_i)
            return outs
        else:
            if isinstance(x, list):
                if len(x)==1:
                    x = x[0]
                else:
                    raise ValueError(f'x is {type(x)}, of length {len(x)}')
            x = self.project(x)
            return x




if __name__ == '__main__':
    # example
    feats1 = torch.rand(size=(2, 1024, 60, 120)).float()
    feats0 = torch.rand(size=(2, 2048, 60, 120)).float()

    # proj = Projector({'mlp': [[1,-1, 1], [1, 256, 1]], 'c_in': [512,512,1024,1024], 'd': 128, 'use_bn': True})

    proj = Projector({'mlp': [[1,-1, 1], [1, 256, 1]], 'c_in': 2048, 'd': 128, "trans": True, "heads":1, 'use_bn': True})

    # projected_feats = proj(([feats0]*2 )+([feats1]*2))
    p = proj(feats0)
    # p_sa = SelfAttention(dim=p.shape[1])(p)
    print(p.shape)

    # print(projected_feats.shape)

    # for v, par in proj.named_parameters():
    #     if par.requires_grad:
    #         print(v, par.data.shape, par.requires_grad)
    # d = proj.state_dict()
    # print(d)
