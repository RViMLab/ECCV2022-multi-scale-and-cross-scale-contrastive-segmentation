import torch
import matplotlib.pyplot as plt
from tsne_torch import TorchTSNE as TSNE
from .utils import to_numpy
import pathlib

def test():
    f = torch.rand(size=(100, 128))
    f_ = TSNE(n_components=2, perplexity=30, n_iter=1000, verbose=True).fit_transform(f)
    l = f_.tolist()
    x,y = zip(*l)
    plt.scatter(x,y)


class TsneMAnager():
    def __init__(self, dataset, n_classes, feat_dim, run_id=None, scale=4):
        self.dataset = dataset
        self.n_classes = n_classes
        self.feats_per_class = 1000
        self.feat_dim = feat_dim
        self.feats = []#torch.zeros(size=(self.n_classes, self.feats_per_class, self.feat_dim))
        self.labels = [] # class id per element of self.feats
        self.counts = [0] * self.n_classes
        self.scale = scale
        self.run_id = run_id if run_id is not None else 'tsne'

        # tsne settings
        self.perplexity = 30
        self.iters = 2000

    def accumulate(self, feats, labels):
        n, h, w = labels.shape
        assert n == 1
        lbl_down = torch.nn.functional.interpolate(labels.unsqueeze(1).float(), (h // self.scale, w // self.scale),
                                                   mode='nearest').long()
        _, _, h, w = lbl_down.shape
        lbl_down = lbl_down.view(-1)
        # feats1 = feats.view(self.feat_dim, h*w)
        feats = feats.squeeze().view(self.feat_dim, -1)  # self.feat_dim, h*w
        if self.dataset == 'CITYSCAPES':
            for cl in range(self.n_classes):
                views_per_class = self.feats_per_class // 500 if cl < 15 else 10
                if self.counts[cl] < self.feats_per_class:
                    indices_from_cl = (lbl_down == cl).nonzero().squeeze()
                    if len(indices_from_cl.shape) > 0:
                        random_permutation = torch.randperm(indices_from_cl.shape[0])
                        this_views_per_class = min(views_per_class, indices_from_cl.shape[0])
                        if this_views_per_class > 0:
                            sampled_indices_from_cl = indices_from_cl[random_permutation[:this_views_per_class]]
                            self.feats.append(feats[:, sampled_indices_from_cl].T)
                            self.labels += [cl] * this_views_per_class  # class id per element of self.feats
                            self.counts[cl] += this_views_per_class
                        # print(f'class {cl} added {this_views_per_class} {len(indices_from_cl)} feats resulting in  counts {self.counts[cl]}')
                else:
                    print(f'class {cl} with counts {self.counts[cl]} is done')
        else:
            raise NotImplementedError()

    def compute(self, log_dir):
        f = torch.cat(self.feats)
        f_tsne = TSNE(n_components=2, perplexity=self.perplexity, n_iter=self.iters, verbose=True).fit_transform(f)
        l = f_tsne.tolist()
        x, y = zip(*l)
        # for colours look here  https://matplotlib.org/3.5.0/gallery/color/named_colors.html
        cmap = {0: "red", 1: "green", 2: "blue", 3: "yellow", 4: "pink", 5: "black", 6: "orange", 7: "purple",
               8: "beige", 9: "brown", 10: "gray", 11: "cyan", 12: "magenta", 13: "hotpink", 14: "darkviolet", 15: "mediumblue",
               16: "lightsteelblue", 17: "gold", 18: "maroon"}
        colors = [cmap[l] for l in self.labels]
        fig = plt.scatter(x, y, c=colors, label=self.labels)
        plt.savefig(str(pathlib.Path(log_dir)/pathlib.Path(
            f'{self.run_id}_perp-{self.perplexity}_its-{self.iters}_feats-per-class-{self.feats_per_class}_scale{self.scale}.png')))
        print(f'counts: {[(i, c) for i, c in enumerate(self.counts)]}')
        return f_tsne



# def get_tsne_embedddings_ms(feats_ms, labels, scale, dataset):
#     n, h, w = labels.shape
#     assert n == 1
#     lbl_down = torch.nn.functional.interpolate(labels.unsqueeze(1).float(), (h//scale, w//scale), mode='nearest').long()
#     assert isinstance(feats_ms, list)
#     if isinstance(feats_ms, list) or isinstance(feats_ms, tuple):
#         for f in feats_ms:
#             get_tsne_embedddings(f, labels, scale, dataset)

#
#
# def get_tsne_embedddings(feats, labels, scale, dataset):
#     print(feats.shape, labels.shape)
#     n, h, w = labels.shape
#     assert n == 1
#     lbl_down = torch.nn.functional.interpolate(labels.unsqueeze(1).float(), (h//scale, w//scale), mode='nearest').long()
#     c = feats.shape[1]  # feature space dimensionality
#     feats = feats.view(h*w, c)
#     lbl_down = lbl_down.view(h*w)
#     if dataset == 'CITYSCAPES':
#         print('computing tsne for CITYSCAPES')
#
#     return 0

