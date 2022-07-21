import numpy as np
import cv2
from PIL import Image
import random
from utils.defaults import DATASETS_INFO


class PadNP(object):
    """Numpy-based function that pads an array by a fixed amount using np.pad()"""
    def __init__(self, ver: tuple, hor: tuple, padding_mode: str):
        self.ver_pad = ver
        self.hor_pad = hor
        self.padding_mode = padding_mode

    def __call__(self, arr: np.ndarray):
        pad_width = (self.ver_pad, self.hor_pad)
        if arr.ndim == 3:
            pad_width += ((0, 0),)
        arr_padded = np.pad(arr, pad_width=pad_width, mode=self.padding_mode)
        return arr_padded


class CropNP(object):
    """Numpy-based function that randomly crops img & lbl to size"""

    def __init__(self,
                 size: float,
                 crop_mode: str = 'random',
                 experiment: int = 1,
                 dataset: str = 'CADIS'):
        self.size = size
        self.crop_mode = crop_mode
        self.experiment = experiment
        self.num_classes = len(DATASETS_INFO[dataset].CLASS_INFO[self.experiment][1])
        class_sums = np.zeros(self.num_classes, 'f')
        if dataset == 'CITYSCAPES':
            raise ValueError('CropNP will not work for ctscapes as currently DATASET_INFO DOES NOT CONTAIN CLASS_SUMS')
        # TODO this loop will not work for ctscapes as currently DATASET_INFO DOES NOT CONTAIN CLASS_SUMS
        for i in range(self.num_classes):
            lookup_class = i if i != 17 else 255
            class_sums[i] = np.sum(np.array(DATASETS_INFO[dataset].CLASS_SUMS)[
                                       DATASETS_INFO[dataset].CLASS_INFO[self.experiment][0][lookup_class]])
        self.class_frequencies = class_sums / np.sum(class_sums)

    def __call__(self, arrs) -> tuple:
        # def __call__(self, img, lbl, metadata=None) -> tuple:
        img = arrs[0]
        lbl = arrs[1]
        metadata = arrs[2] if len(arrs) == 3 else None
        if isinstance(img, Image.Image):
            img = np.array(img)
            lbl = np.array(lbl)
        img_h, img_w = img.shape[0], img.shape[1]
        actual_crop_px = int(32*((self.size * img_h) // 32))  # so the patches work in the network
        if actual_crop_px >= img_h or actual_crop_px >= img_w:  # Crop larger than one of dims, cut down
            actual_crop_px = min(img_h, img_w)
        if self.crop_mode == 'random':
            v_min, v_max = 0, img_h - actual_crop_px
            h_min, h_max = 0, img_w - actual_crop_px
            # Calculate upper left corner
            v = random.randint(v_min, v_max) if v_max > v_min else v_min
            h = random.randint(h_min, h_max) if h_max > h_min else h_min
        elif self.crop_mode == 'freq':
            # ___________
            # |    margin = actual_crop_px // 2
            # |    ______
            # |    |
            margin = actual_crop_px // 2
            px_probs = 1 / self.class_frequencies[lbl][margin:img_h - margin, margin:img_h - margin]
            px_probs /= np.sum(px_probs)
            px_selection = random.choices(list(range(px_probs.size)), weights=px_probs.flatten().tolist(), k=1)[0]
            v = px_selection // px_probs.shape[1]
            h = px_selection % px_probs.shape[1]
            # in theory: + margin to correct for offset, but actually we want the left upper corner, so - margin again
        else:
            raise ValueError("Crop mode '{}' not recognised.".format(self.crop_mode))
        lbl_cropped = lbl[v:v + actual_crop_px, h:h + actual_crop_px]
        img_cropped = img[v:v + actual_crop_px, h:h + actual_crop_px]

        if metadata:
            metadata.update({
                'crop_offsets': [v, h],
                'crop_centres': [v + actual_crop_px // 2, h + actual_crop_px // 2],
                'crop_size': actual_crop_px
            })
            return img_cropped, lbl_cropped, metadata
        else:
            return img_cropped, lbl_cropped


class FlipNP(object):
    """Numpy-based function that randomly flips img & lbl the same way, with (ver_prob, hor_prob), default (0, .5)"""
    def __init__(self, probability: tuple = None):
        self.probability = (0, .5) if probability is None else probability  # ver, hor probability

    def __call__(self, arrs: tuple) -> tuple:
        # def __call__(self, img, lbl, metadata=None) -> tuple:
        img = arrs[0]
        lbl = arrs[1]
        metadata = arrs[2] if len(arrs) == 3 else None
        flip_dims = []
        if np.random.random() < self.probability[0]:
            flip_dims.append(-2)
            img = np.flip(img, axis=0)
            lbl = np.flip(lbl, axis=0)
        if np.random.random() < self.probability[1]:
            img = np.flip(img, axis=1)
            lbl = np.flip(lbl, axis=1)
            flip_dims.append(-1)
        if metadata:
            metadata['flip_dims'] = np.sum(flip_dims)  # workaround to get it passed properly: -1, -2, or -3 if both
            return img.copy(), lbl.copy(), metadata
        else:
            return img.copy(), lbl.copy()



class AffineNP(object):
    def __init__(self, num_classes: int, crop_to_fit: bool = None,
                 rotation: int = None, rot_centre_offset: tuple = None,
                 shift: int = None,
                 shear: tuple = None, shear_centre_offset: tuple = None):
        self.num_classes = num_classes
        self.crop_to_fit = True if crop_to_fit is None else crop_to_fit
        self.rotation = 10 if rotation is None else rotation  # maximum rotation in either direction
        self.rot_centre_offset = (.25, .25) if rot_centre_offset is None else rot_centre_offset
        # Offset of rotation centre from image centre, as fraction of H / W
        self.shift = .1 if shift is None else shift  # maximum shift vertically, horizontally as fraction of H / W
        self.shear = (.1, .1) if shear is None else shear  # Ver / hor shearing as fraction of image
        self.shear_centre_offset = (.25, .25) if shear_centre_offset is None else shear_centre_offset
        # Offset of shear centre from image centre, as fraction of H / W

    def __call__(self, arr1, arr2, metadata=None):
        arrs = [arr1, arr2]
        rot_vals = self.get_rot_vals(arrs[0])
        rot_matrix = get_rot_matrix(rot_vals)
        shift_vals = self.get_shift_vals(arrs[0])
        shift_matrix = get_shift_matrix(shift_vals)
        shear_vals = self.get_shear_vals(arrs[0])
        shear_matrix = get_shear_matrix(shear_vals)
        matrix = shift_matrix @ rot_matrix @ shear_matrix
        mask = np.ones_like(arrs[1])
        src = np.concatenate((arrs[0], mask[..., np.newaxis], to_one_hot_np(arrs[1], self.num_classes)), axis=2)
        warped = cv2.warpPerspective(src, matrix, (src.shape[1] * 2, src.shape[0] * 2))
        img = np.round(warped[..., :3]).astype('uint8')
        mask_warped = warped[..., 3]
        if self.crop_to_fit:
            rect = rect_from_mask(mask_warped, dims=mask.shape, scale=16)
            img = cv2.resize(img[rect[2]:rect[3], rect[0]:rect[1]], mask.shape[::-1])
            lbl = np.argmax(cv2.resize(warped[..., 4:][rect[2]:rect[3], rect[0]:rect[1]], mask.shape[::-1]), axis=2)
        else:
            lbl = np.argmax(warped[..., 4:], axis=2)

        if metadata:
            metadata = arrs[-1]
            metadata['affine_rot_vals'] = rot_vals
            metadata['affine_shift_vals'] = shift_vals
            metadata['affine_shear_vals'] = shear_vals
        return img, lbl, metadata

    def get_shift_vals(self, array):
        shift_ver = int(np.round(array.shape[0] * self.shift * np.random.rand()))
        shift_hor = int(np.round(array.shape[1] * self.shift * np.random.rand()))
        return shift_ver, shift_hor

    def get_rot_vals(self, array):
        rot = self.rotation * (2 * np.random.rand() - 1)
        rot_centre_ver = int(np.round(array.shape[0] * (.5 + self.rot_centre_offset[0] * (2 * np.random.rand() - 1))))
        rot_centre_hor = int(np.round(array.shape[1] * (.5 + self.rot_centre_offset[1] * (2 * np.random.rand() - 1))))
        return rot_centre_ver, rot_centre_hor, rot

    def get_shear_vals(self, array):
        shear_ver = self.shear[0] * (2 * np.random.rand() - 1)
        shear_hor = self.shear[1] * (2 * np.random.rand() - 1)
        shear_centre_ver = int(np.round(array.shape[0] * (.5 + self.shear_centre_offset[0] * (2 * np.random.rand() - 1))))
        shear_centre_hor = int(np.round(array.shape[1] * (.5 + self.shear_centre_offset[1] * (2 * np.random.rand() - 1))))
        return shear_centre_ver, shear_centre_hor, shear_ver, shear_hor


def get_shift_matrix(shift_vals: tuple):
    matrix = np.identity(3)
    matrix[0:2, 2] = shift_vals[1], shift_vals[0]
    return matrix


def get_rot_matrix(rot_vals: tuple):
    matrix = np.identity(3)
    rot = np.radians(rot_vals[2])
    translation_matrix_1 = get_shift_matrix((-rot_vals[0], -rot_vals[1]))
    translation_matrix_2 = get_shift_matrix(rot_vals[:2])
    matrix[0:2, 0:2] = [[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]
    matrix = translation_matrix_2 @ matrix @ translation_matrix_1
    return matrix


def get_shear_matrix(shear_vals: tuple):
    translation_matrix_1 = get_shift_matrix((-shear_vals[0], -shear_vals[1]))
    translation_matrix_2 = get_shift_matrix(shear_vals[:2])
    matrix = np.identity(3)
    matrix[1, 0] = shear_vals[2]
    matrix[0, 1] = shear_vals[3]
    matrix = translation_matrix_2 @ matrix @ translation_matrix_1
    return matrix


def to_one_hot_np(array, num_classes):
    res = np.eye(num_classes)[np.array(array).reshape(-1)]
    one_hot = res.reshape(*array.shape, num_classes)
    return one_hot


def rect_from_mask(mask, dims, scale):
    """From mask (0 invalid, 1 valid) and given dims find largest rectangle ratio ver_dim:hor_dim inscribed in mask"""
    # https://gis.stackexchange.com/questions/59215/how-to-find-the-maximum-area-rectangle-inside-a-convex-polygon
    # Above says: if rectangle not square, three points will be on boundary. This means that if we start at one point,
    #   find the next two corner points, follow to the natural 4th and the line hits the boundary so that the resulting
    #   actual rectangle would be smaller, this means that adjusting to get to a better rectangle would mean two
    #   vertices are inside the boundary --> not maximal anyway.
    #   There can be exceptions for this if the line to the corner points from the first points is on a boundary, but in
    #   that case we would still detect the actual maximal rectangle later anyway, when we systematically go through all
    #   other boundary points available as starting points.
    mask = np.round(mask).astype('i')  # Mask cleanup
    # noinspection PyBroadException
    try:
        v1_r, v2_r, h1_r, h2_r, f_r = _fit_rect_single_side(mask, dims, scale)  # Starting from left
        h1_c, h2_c, v1_c, v2_c, f_c = _fit_rect_single_side(np.transpose(mask), dims[::-1], scale)  # Starting from top
        if f_r > f_c:
            h1, h2, v1, v2 = h1_r, h2_r, v1_r, v2_r
        else:
            h1, h2, v1, v2 = h1_c, h2_c, v1_c, v2_c
        return h1, h2, v1, v2
    except Exception:
        # Fallback in case the mask hasn't loaded properly,
        # or an error occurs in _fit_rect_single_side (see screenshot 2020_07_02 in OneDrive docs)
        print("                     Warning: rect_from_mask fallback used")
        return 0, mask.shape[1] - 1, 0, mask.shape[0] - 1


def _fit_rect_single_side(mask, dims, scale):
    """Drawings see 29/05/20, notebook 2"""

    assert scale % 2 == 0
    mask = mask[::scale, ::scale]
    dims = [dims[0]//scale, dims[1]//scale]

    r_sum = np.sum(mask, axis=1)

    # Step 1: get pt1 / pt2 (left-most / right-most leading points), and tops / bots (upper / lower boundary pixels)
    pt1 = np.stack((np.arange(mask.shape[0]), (mask != 0).argmax(axis=1)), axis=-1)[r_sum > 0]
    pt2 = np.stack((np.arange(mask.shape[0]), (mask[:, ::-1] != 0).argmax(axis=1)), axis=-1)[r_sum > 0]
    pt2[:, 1] = mask.shape[1] - pt2[:, 1] - 1  # correct for flipping
    tops = np.stack(((mask != 0).argmax(axis=0), np.arange(mask.shape[1])), axis=-1)
    bots = np.stack(((mask[::-1, :] != 0).argmax(axis=0), np.arange(mask.shape[1])), axis=-1)
    bots[:, 0] = mask.shape[0] - bots[:, 0] - 1  # correct for flipping

    # Step 2: construct array [num_pt * num_pt * 2=(ver, hor)], for each point comb the vertical / horizontal distance
    dists = (np.abs(np.expand_dims(pt1, axis=1) - np.expand_dims(pt2, axis=0))).astype('f')

    # Step 3: construct 2 arrays [num_pt * 2(ver up, ver down)], for each point how far up / down there is 'space'
    space1 = np.stack((pt1[:, 0] - tops[pt1[:, 1], 0], bots[pt1[:, 1], 0] - pt1[:, 0]), axis=-1)
    space2 = np.stack((pt2[:, 0] - tops[pt2[:, 1], 0], bots[pt2[:, 1], 0] - pt2[:, 0]), axis=-1)

    # Step 4: if dist vertically 0, set all vertical dists to max of the min of space either side of the two points
    max_min_space = np.max(np.minimum(space1, space2), axis=-1)
    dists[np.identity(len(max_min_space)) > 0, 0] = max_min_space

    # Step 5: set all dists to 0 where the pt1/pt2 combination does not work
    pt1_lims = np.stack((tops[pt1[:, 1], 0], bots[pt1[:, 1], 0]), axis=-1)
    pt2_lims = np.stack((tops[pt2[:, 1], 0], bots[pt2[:, 1], 0]), axis=-1)
    pt1_within = np.greater_equal(pt1[:, 0, np.newaxis], pt2_lims[np.newaxis, :, 0]) &\
        np.less_equal(pt1[:, 0, np.newaxis], pt2_lims[np.newaxis, :, 1])
    pt2_within = np.greater_equal(pt2[:, 0, np.newaxis], pt1_lims[np.newaxis, :, 0]) & \
        np.less_equal(pt2[:, 0, np.newaxis], pt1_lims[np.newaxis, :, 1])
    within = pt1_within & np.transpose(pt2_within)
    dists[~within] = 0

    # Step 6: convert dists into fractions of the input dims, find dir-pair-wise minimum, and overall maximum
    fractions = np.copy(dists) + 1
    fractions[..., 0] /= dims[0]
    fractions[..., 1] /= dims[1]
    min_dir_fraction = np.min(fractions, axis=-1)
    max_fraction = np.max(min_dir_fraction)
    idx = np.unravel_index(min_dir_fraction.argmax(), min_dir_fraction.shape[:2])
    dom_dir = np.argmin(dists[idx])
    p1, p2 = pt1[idx[0]], pt2[idx[1]]
    # NOTE: following line should work, but... when it doesn't, it just throws an error and what's the point.
    # assert mask[p1[0], p2[1]] > 0 and mask[p2[0], p1[1]] > 0

    # Step 7: determine v1, h1, v2, h2
    v1, h1 = p1
    h2 = p2[1]
    v2 = None
    if p1[0] == p2[0]:  # two points in one line
        v2_sel = [v1 - dists[idx[0], idx[1], 0], v1 + dists[idx[0], idx[1], 0]]
        for v in v2_sel:
            v = int(v)
            if 0 <= v < mask.shape[0]:
                if mask[v, h1] > 0 and mask[v, h2] > 0:
                    v2 = v
    else:
        v2 = p2[0]
    [v1, v2] = np.sort([v1, v2])

    # Step 8: determine actual borders with correct ratio
    if dom_dir == 0:  # vertical direction correct: v1, v2, h1 unchanged, h2 adjusted
        v_dist = v2 - v1 + 1
        h_dist = np.floor(v_dist * dims[1] / dims[0]).astype('i')
        h2 = h1 + h_dist - 1
    else:  # horizontal direction correct: h1, h2, v1 unchanged, v2 adjusted
        h_dist = h2 - h1 + 1
        v_dist = np.floor(h_dist * dims[0] / dims[1]).astype('i')
        v2 = v1 + v_dist - 1
    # TODO: figure out why this is not pixel perfect?

    v1, v2, h1, h2 = scale*v1, scale*v2, scale*h1, scale*h2

    return v1, v2, h1, h2, max_fraction