import torch
import torch.nn as nn
from utils import DATASETS_INFO, get_rank, printlog
from torch.nn.functional import one_hot
from utils import Logger as Log

def has_inf_or_nan(x):
    return torch.isinf(x).max().item(), torch.isnan(x).max().item()


class DenseContrastiveLossV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experiment = config['experiment']
        self.dataset = config['dataset']
        self.num_all_classes = len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1])
        self.num_real_classes = self.num_all_classes - 1 if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] else self.num_all_classes
        self.ignore_class = (len(DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]) - 1) if 255 in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1] else -1
        self.temperature = config['temperature'] if 'temperature' in config else 0.5
        self.base_temperature = 1.0
        self.min_views_per_class = config['min_views_per_class'] if 'min_views_per_class' in config else 5
        self.label_scaling_mode = config['label_scaling_mode'] if 'label_scaling_mode' in config else 'nn'
        self.cross_scale_contrast = config['cross_scale_contrast'] if 'cross_scale_contrast' in config else False
        self.dominant_mode = 'all'
        self.eps = torch.tensor(1e-10)
        self.metadata = {}
        self.max_views_per_class = config['max_views_per_class'] if 'max_views_per_class' in config else 2500
        self.max_features_total = config['max_features_total'] if 'max_features_total' in config else 10000
        self.log_this_step = False
        self._scale = None

        for class_name in DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][1]:
            self.metadata[class_name] = (0.0, 0.0)  # pos-neg per class
        # self.anchors_per_image = config['anchors_per_image'] if 'anchors_per_image' in config else 50
        # sanity checks
        if self.label_scaling_mode == 'nn':
            # when using nn interpolation to get dominant_class in feature space
            # dominant_mode can only be in 'all' mode
            # there is no notion of entropy or cross entropy in the class_distr so these weighting must be set to False
            # dominant
            assert(self.dominant_mode == 'all'), \
                'cannot use label_scaling_mode: "{}" with dominant_mode: "{}" -' \
                ' only "all" is allowed'.format(self.label_scaling_mode, self.dominant_mode)

    def forward(self, label: torch.Tensor, features: torch.Tensor):
        flag_error = False
        with torch.no_grad():  # Not sure if necessary, but these steps neither are nor need to be differentiable
            scale = int(label.shape[-1] // features.shape[-1])
            class_distribution, dominant_classes = self.get_dist_and_classes(label, scale)
            # example_identification = self.identify_examples(dominant_classes, non_ignored_anchors)
        sampled_features, sampled_labels, flag_error = self.sample_anchors_fast(dominant_classes, features)

        # old
        # sampled_features, sampled_labels, flag_error = self.sample_anchors(dominant_classes, features)
        if flag_error:
            loss = features-features
            loss = loss.mean()
        else:
            loss = self.contrastive_loss(sampled_features, sampled_labels)
        # feature_correlations = self.correlate_features(features)
        # loss = self.calculate_loss(feature_correlations, dominant_classes, example_identification,
        #                            class_distribution, non_ignored_anchors)
        # return loss.mean()
        if self.cross_scale_contrast:
            return loss, sampled_features, sampled_labels, flag_error
        return loss

    def _select_views_per_class(self, min_views, total_cls, cls_in_batch, cls_counts_in_batch):
        if self.max_views_per_class == 1:
            # no cappping to views_per_class
            views_per_class = min_views
        else:
            # capping views_per_class to avoid OOM
            views_per_class = min(min_views, self.max_views_per_class)
            if views_per_class == self.max_views_per_class:
                Log.info(
                    f'\n rank {get_rank()} capping views per class to {self.max_views_per_class},'
                    f' cls_and_counts: {cls_in_batch} {cls_counts_in_batch} ')
                self.log_this_step = True
        if views_per_class * total_cls > self.max_features_total:
            views_per_class = self.max_features_total // total_cls
            printlog(
                f'\n rank {get_rank()}'
                f' capping total features  to {self.max_features_total} total_cls:  {total_cls} '
                f'--> views_per_class:  {views_per_class} ,'
                f'  cls_and_counts: {cls_in_batch} {cls_counts_in_batch}')
            self.log_this_step = True
        return views_per_class

    def sample_anchors_fast(self, dominant_classes, features):
        """
        self.anchors_per_image =
        :param dominant_classes: N-H-W-1
        :param features:  N-C-H-W
        :return: sampled_features: (classes_in_batch, C, views)
                 sampled_labels : (classes_in_batch)
        """
        flag_error = False
        n = dominant_classes.shape[0]  # batch size
        c = features.shape[1]  # feature space dimensionality
        features = features.view(n, c, -1)
        dominant_classes = dominant_classes.view(n, -1)  # flatten   # flatten n,1,h,w --> n,h*w
        skip_ids = []
        cls_in_batch = []  # list of lists each containing classes in an image of the batch
        cls_counts_in_batch = []  # list of lists each containing classes in an image of the batch

        classes_ids = torch.arange(start=0, end=self.num_all_classes, step=1, device=dominant_classes.device)
        compare = dominant_classes.unsqueeze(-1) == classes_ids.unsqueeze(0).unsqueeze(0)# n, hw, 1 == 1, 1, n_c => n,hw,n_c
        cls_counts = compare.sum(1) # n, n_c

        present_inds = torch.where(cls_counts[:, :-1] >= self.min_views_per_class) # ([0,...,n-1], [prese   nt class ids])
        batch_inds, cls_in_batch = present_inds

        min_views = torch.min(cls_counts[present_inds])
        total_cls = cls_in_batch.shape[0]

        views_per_class = self._select_views_per_class(min_views, total_cls, cls_in_batch, cls_counts_in_batch)
        sampled_features = torch.zeros((total_cls, c, views_per_class), dtype=torch.float).cuda()
        sampled_labels = torch.zeros(total_cls, dtype=torch.float).cuda()

        for i in range(total_cls):
            # print(batch_inds[i], cls_in_batch[i])
            indices_from_cl_fast = compare[batch_inds[i], :, cls_in_batch[i]].nonzero().squeeze()
            # indices_from_cl = (dominant_classes[batch_inds[i]] == cls_in_batch[i]).nonzero().squeeze()
            random_permutation = torch.randperm(indices_from_cl_fast.shape[0]).cuda()
            sampled_indices_from_cl = indices_from_cl_fast[random_permutation[:views_per_class]]
            sampled_features[i] = features[batch_inds[i], :, sampled_indices_from_cl]
            sampled_labels[i] = cls_in_batch[i]

        return sampled_features, sampled_labels, flag_error


    def sample_anchors(self, dominant_classes, features):
        """
        self.anchors_per_image =
        :param dominant_classes: N-H-W-1
        :param features:  N-C-H-W
        :return: sampled_features: (classes_in_batch, C, views)
                 sampled_labels : (classes_in_batch)
        """
        flag_error = False
        n = dominant_classes.shape[0]  # batch size
        c = features.shape[1]  # feature space dimensionality
        features = features.view(n, c, -1)
        dominant_classes = dominant_classes.view(n, -1)
        skip_ids = []
        cls_in_batch = []  # list of lists each containing classes in an image of the batch
        cls_counts_in_batch = []  # list of lists each containing classes in an image of the batch
        total_cls = 0  # classes in batch (non-unique)
        min_views = 10000
        for i in range(n):
            y_i = dominant_classes[i].squeeze()
            # cls_in_y_i = torch.unique(y_i, return_counts=True)
            # classes in i-th image of the batch
            cls_in_y_i, cls_counts_in_y_i = torch.unique(y_i, return_counts=True)
            # filter out ignore_class and classes with few views
            cls_and_counts = [(cl.item(), cl_count.item()) for cl, cl_count in zip(cls_in_y_i, cls_counts_in_y_i)
                              if cl != self.ignore_class
                              and cl_count.item() >= self.min_views_per_class]
            if len(cls_and_counts) == 0:
                # only ignore class in labels
                skip_ids.append(i)
            else:
                cls_and_counts = [x for x in zip(*cls_and_counts)]
                cls_in_y_i = list(cls_and_counts[0])
                cls_counts_in_y_i = list(cls_and_counts[1])
                # keep track of smallest class count
                min_views_current = min(cls_counts_in_y_i)
                if min_views_current < min_views:
                    min_views = min_views_current
                total_cls += len(cls_in_y_i)


            cls_counts_in_batch.append(cls_counts_in_y_i)
            cls_in_batch.append(cls_in_y_i)

        if len(skip_ids) == n:
            flag_error = True
            Log.info(f'\n rank {get_rank()} cls_and_counts : {cls_in_batch} skipping this batch')

        # select how many samples per class (with repetition) will be sampled
        views_per_class = self._select_views_per_class(min_views, total_cls, cls_in_batch, cls_counts_in_batch)

        # tensors to be populated with anchors
        sampled_features = torch.zeros((total_cls, c, views_per_class), dtype=torch.float).cuda()
        sampled_labels = torch.zeros(total_cls, dtype=torch.float).cuda()
        ind = 0
        for i in range(n):
            if i in skip_ids:
                continue
            cls_in_y_i = cls_in_batch[i]  # classes in image
            y_i = dominant_classes[i].squeeze()
            for cl in cls_in_y_i:
                indices_from_cl = (y_i == cl).nonzero().squeeze()
                random_permutation = torch.randperm(indices_from_cl.shape[0])
                sampled_indices_from_cl = indices_from_cl[random_permutation[:views_per_class]]
                sampled_features[ind] = features[i, :, sampled_indices_from_cl]
                sampled_labels[ind] = cl
                # print(ind, cl, indices_from_cl.shape[0], sampled_indices_from_cl.shape[0], views_per_class)
                ind += 1
        return sampled_features, sampled_labels, flag_error

    def contrastive_loss(self, feats, labels):
        """
        :param feats: T-C-V
                      T: classes in batch (with repetition), which can be thought of as the number of anchors
                      C: feature space dimensionality
                      V: views per class (i.e samples from each class),
                       which can be thought of as the number of views per anchor
        :param labels: T
        :return: loss
        """
        # prepare feats
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)  # L2 normalization
        feats = feats.transpose(dim0=1, dim1=2)  # feats are T-V-C
        num_anchors, views_per_anchor, c = feats.shape  # get T, V, C
        labels = labels.contiguous().view(-1, 1)  # labels are T-1

        # print( f'rank: {get_rank()} -- classes {num_anchors} v_per_class {views_per_anchor} total_anchors = {num_anchors * views_per_anchor}')
        # feats_flat = torch.cat(torch.unbind(feats, dim=1), dim=0)  # feats_flat is V*T-C
        # dot_product = torch.div(torch.matmul(feats_flat, torch.transpose(feats_flat, 0, 1)), self.temperature)
        # # dot_product # V*T-C @ C-V*T = V*T-V*T
        #
        # mask, pos_mask, neg_mask = self.get_masks(labels, num_anchors, views_per_anchor)
        # loss = self.compute(pos_mask, neg_mask, dot_product)
        # print(loss)

        # modifying to more intuitive version
        labels_ = labels.repeat(1, views_per_anchor)  # labels are T-V
        labels_ = labels_.view(-1, 1)  # labels are T*V-1
        pos_mask2, neg_mask2 = self.get_masks2(labels_, num_anchors, views_per_anchor)
        feats_flat = feats.contiguous().view(-1, c)  # feats_flat is T*V-C
        dot_product = torch.div(torch.matmul(feats_flat, torch.transpose(feats_flat, 0, 1)), self.temperature)
        loss2 = self.get_loss(pos_mask2, neg_mask2, dot_product)
        # print(loss2)
        return loss2

    @staticmethod
    def get_masks(labels, num_anchors, views_per_anchor):
        """
        :param labels: T*V-1
        :param num_anchors: T
        :param views_per_anchor: V
        :return: mask, pos_maks,
        """
        # extract mask indicating same class samples
        mask = torch.eq(labels, torch.transpose(labels, 0, 1)).float()  # mask T-T
        mask = mask.repeat(views_per_anchor, views_per_anchor)  # mask V*T-V*T
        neg_mask = 1 - mask  # indicator of negatives
        # set diagonal mask elements to zero -- self-similarities
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(num_anchors * views_per_anchor).view(-1, 1).cuda(),
                                                     0)
        pos_mask = mask * logits_mask  # indicator of positives
        return pos_mask, neg_mask

    @staticmethod
    def get_masks2(labels, num_anchors, views_per_anchor):
        """
        takes flattened labels and identifies pos/neg of each anchor
        :param labels: T*V-1
        :param num_anchors: T
        :param views_per_anchor: V
        :return: mask, pos_maks,
        """
        # extract mask indicating same class samples
        mask = torch.eq(labels, torch.transpose(labels, 0, 1)).float()  # mask T-T
        neg_mask = 1 - mask  # indicator of negatives
        # set diagonal mask elements to zero -- self-similarities
        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(num_anchors * views_per_anchor).view(-1, 1).cuda(),
                                                     0)
        pos_mask = mask * logits_mask  # indicator of positives
        return pos_mask, neg_mask

    def get_loss(self, pos, neg, dot):
        """
        :param pos: V*T-V*T
        :param neg: V*T-V*T
        :param dot: V*T-V*T
        :return:
        """
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = dot  # - logits_max.detach()

        neg_logits = torch.exp(logits) * neg
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)
        # print('exp_logits ', has_inf_or_nan(exp_logits))
        log_prob = logits - torch.log(exp_logits + neg_logits)
        # print('log_prob ', has_inf_or_nan(log_prob))

        mean_log_prob_pos = (pos * log_prob).sum(1) / pos.sum(1)  # normalize by positives
        # print('\npositives: {} \nnegatives {}'.format(pos.sum(1), neg.sum(1)))
        # print('mean_log_prob_pos ', has_inf_or_nan(mean_log_prob_pos))
        loss = - mean_log_prob_pos
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        loss = loss.mean()
        # print('loss.mean() ', has_inf_or_nan(loss))
        # print('loss {}'.format(loss))
        if has_inf_or_nan(loss)[0] or has_inf_or_nan(loss)[1]:
            printlog(f'\n rank {get_rank()} inf found in loss with positives {pos.sum(1)} and Negatives {neg.sum(1)}')
        return loss

    def get_dist_and_classes(self, label: torch.Tensor, scale: int) -> torch.Tensor:
        """Determines the distribution of the classes in each scale*scale patch of the ground truth label N-H-W,
        for given experiment, returning class_distribution as N-C-H//scale-W//scale tensor. Also determines dominant
        classes in each patch of the ground truth label N-H-W, based on the class distribution. Output is
        N-C-H//scale-W//scale where C might be 1 (just one dominant class) or more.

        If label_scaling_mode == 'nn' peforms nearest neighbour interpolation on the label without one_hot encoding and
        returns N-1-H//scale-W//scale
        """
        n, h, w = label.shape
        self._scale = scale
        if self.label_scaling_mode == 'nn':
            lbl_down = torch.nn.functional.interpolate(label.unsqueeze(1).float(), (h//scale, w//scale), mode='nearest')
            # non_ignored_anchors = non_ignored_anchors = (lbl_down != self.num_real_classes).view(n, h//scale * w//scale)
            return lbl_down.long(), lbl_down.long()

        elif self.label_scaling_mode == 'avg_pool':
            lbl_one_hot = one_hot(label.to(torch.int64), self.num_all_classes).permute(dims=[0, 3, 1, 2])
            class_distribution = torch.nn.AvgPool2d(kernel_size=scale)(lbl_one_hot.float())
            # class_distribution is:    N-C|all-H//scale-W//scale
            dominant_classes = self.get_dominant_classes(class_distribution)
            # dominant_classes is:      N-1-H//scale-W//scale
            non_ignored_anchors = (dominant_classes != self.num_real_classes).view(n, 1, h//scale, w//scale)
            # non_ignored_anchors is:   N-1-H//scale-W//scale
            # Note: if e.g. exp = 2, then we're looking for dom_class == num_all_classes - 1 (18 - 1 = 17), which is
            #   the same as num_real_classes (= 17), because num_real_classes is num_all_classes - 1
            #   (analogous when exp = 3)
            if self.experiment in [2, 3]:  # Need to cut and re-normalise the class_distribution
                class_distribution = class_distribution[:, :self.num_real_classes]
                # class_distribution is:    N-C|real-H//scale-W//scale
                norm_sum = torch.sum(class_distribution, dim=1, keepdim=True)
                norm_sum[norm_sum == 0] = 1
                class_distribution /= norm_sum
                class_distribution[~non_ignored_anchors.repeat(1, self.num_real_classes, 1, 1)] = \
                    1 / self.num_real_classes
                # NOTE: set class_distribution where ignored anchors are to default 1 / num_real_classes to avoid any
                #   issues with zeros during the loss calculation - eventually ignored anyway
            return class_distribution, dominant_classes, non_ignored_anchors.view(n, h//scale * w//scale)

    def get_dominant_classes(self, class_distribution: torch.Tensor, mode: str = None) -> torch.Tensor:
        """Determines dominant classes in each scale*scale patch of the ground truth label N-H-W, based on the N-C-H-W
        class distribution passed. Output is N-C-H-W where C might be 1 (just one dominant class) or more"""
        mode = self.dominant_mode if mode is None else mode
        # class_distribution is N-C-H*W
        if mode == 'all':
            dom_classes = torch.argmax(class_distribution, dim=1).unsqueeze(1)  # dom_classes is N-H-W
        elif mode in ['instruments', 'rare']:
            class_selection = DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][2][self.dominant_mode]
            dom_classes = torch.argmax(class_distribution[:, class_selection], dim=1).unsqueeze(1)
            cond_not_satisfied = torch.gather(class_distribution, dim=1, index=dom_classes) < self.dominant_thresh
            dom_classes[cond_not_satisfied] = self.get_dominant_classes(class_distribution, 'all')[cond_not_satisfied]
        # elif self.dominant_mode == 'multiple':
        #     class_selection = DATASETS_INFO[self.dataset].CLASS_INFO[self.experiment][2]['instruments']
        #     dom_classes = None
        else:
            raise ValueError("Mode '{}' not recognised".format(self.dominant_mode))
        return dom_classes


