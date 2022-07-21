from typing import Any


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


categories_exp0 = {
    'flat': [1, 2],
}

categories_exp1 = {
    'flat': [1, 2],
}

class_remapping_exp0 = {
    0:[255],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7],
    8: [8],
    9: [9],
    10: [10],
    11: [11],
    12: [12],
    13: [13],
    14: [14],
    15: [15],
    16: [16],
    17: [17],
    18: [18],
    19: [19],
    20: [20],
    21: [21],
    22: [22],
    23: [23],
    24: [24],
    25: [25],
    26: [26],
    27: [27],
    28: [28],
    29: [29],
    30: [30],
    31: [31],
    32: [32],
    33: [33],
    34: [34],
    35: [35],
    36: [36],
    37: [37],
    38: [38],
    39: [39],
    40: [40],
    41: [41],
    42: [42],
    43: [43],
    44: [44],
    45: [45],
    46: [46],
    47: [47],
    48: [48],
    49: [49],
    50: [50],
    51: [51],
    52: [52],
    53: [53],
    54: [54],
    55: [55],
    56: [56],
    57: [57],
    58: [58],
    59: [59]
}
#
# CLASSES = ('background', 'aeroplane', 'bag', 'bed', 'bedclothes', 'bench',
#            'bicycle', 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
#            'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
#            'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
#            'floor', 'flower', 'food', 'grass', 'ground', 'horse',
#            'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person',
#            'plate', 'platform', 'pottedplant', 'road', 'rock', 'sheep',
#            'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa', 'table',
#            'track', 'train', 'tree', 'truck', 'tvmonitor', 'wall', 'water',
#            'window', 'wood')
#
# PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
#            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
#            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
#            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
#            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
#            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
#            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
#            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
#            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
#            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
#            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
#            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
#            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
#            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
#            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]]

classes_exp0 = {
    0: "background",
    1: "aeroplane",
    2: "bag",
    3: "bed",
    4: "bedclothes",
    5: "bench",
    6: "bicycle",
    7: "bird",
    8: "boat",
    9: "book",
    10: "bottle",
    11: "building",
    12: "bus",
    13: "cabinet",
    14: "car",
    15: "cat",
    16: "ceiling",
    17: "chair",
    18: "cloth",
    19: "computer",
    20: "cow",
    21: "cup",
    22: "curtain",
    23: "dog",
    24: "door",
    25: "fence",
    26: "floor",
    27: "flower",
    28: "food",
    29: "grass",
    30: "ground",
    31: "horse",
    32: "keyboard",
    33: "light",
    34: "motorbike",
    35: "mountain",
    36: "mouse",
    37: "person",
    38: "plate",
    39: "platform",
    40: "pottedplant",
    41: "road",
    42: "rock",
    43: "sheep",
    44: "shelves",
    45: "sidewalk",
    46: "sign",
    47: "sky",
    48: "snow",
    49: "sofa",
    50: "table",
    51: "track",
    52: "train",
    53: "tree",
    54: "truck",
    55: "tvmonitor",
    56: "wall",
    57: "water",
    58: "window",
    59: "wood"
}


class_remapping_exp1 = {
    255: [0],
    0: [1],
    1: [2],
    2: [3],
    3: [4],
    4: [5],
    5: [6],
    6: [7],
    7: [8],
    8: [9],
    9: [10],
    10: [11],
    11: [12],
    12: [13],
    13: [14],
    14: [15],
    15: [16],
    16: [17],
    17: [18],
    18: [19],
    19: [20],
    20: [21],
    21: [22],
    22: [23],
    23: [24],
    24: [25],
    25: [26],
    26: [27],
    27: [28],
    28: [29],
    29: [30],
    30: [31],
    31: [32],
    32: [33],
    33: [34],
    34: [35],
    35: [36],
    36: [37],
    37: [38],
    38: [39],
    39: [40],
    40: [41],
    41: [42],
    42: [43],
    43: [44],
    44: [45],
    45: [46],
    46: [47],
    47: [48],
    48: [49],
    49: [50],
    50: [51],
    51: [52],
    52: [53],
    53: [54],
    54: [55],
    55: [56],
    56: [57],
    57: [58],
    58: [59]
}


classes_exp1 = {
    255: "background",
    0: "aeroplane",
    1: "bag",
    2: "bed",
    3: "bedclothes",
    4: "bench",
    5: "bicycle",
    6: "bird",
    7: "boat",
    8: "book",
    9: "bottle",
    10: "building",
    11: "bus",
    12: "cabinet",
    13: "car",
    14: "cat",
    15: "ceiling",
    16: "chair",
    17: "cloth",
    18: "computer",
    19: "cow",
    20: "cup",
    21: "curtain",
    22: "dog",
    23: "door",
    24: "fence",
    25: "floor",
    26: "flower",
    27: "food",
    28: "grass",
    29: "ground",
    30: "horse",
    31: "keyboard",
    32: "light",
    33: "motorbike",
    34: "mountain",
    35: "mouse",
    36: "person",
    37: "plate",
    38: "platform",
    39: "pottedplant",
    40: "road",
    41: "rock",
    42: "sheep",
    43: "shelves",
    44: "sidewalk",
    45: "sign",
    46: "sky",
    47: "snow",
    48: "sofa",
    49: "table",
    50: "track",
    51: "train",
    52: "tree",
    53: "truck",
    54: "tvmonitor",
    55: "wall",
    56: "water",
    57: "window",
    58: "wood"
}


CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1]
]

CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]]

PASCALC_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES)


def label_sanity_check(root=None):
    import cv2
    import warnings
    import pathlib
    import numpy as np
    warning = 0
    warning_msg = []
    if root == None:
        root = pathlib.Path(r"C:\Users\Theodoros Pissas\Documents\tresorit\PASCALC\val\label/")
    for path_to_label in root.glob('**/*.PNG'):
        i = cv2.imread(str(path_to_label))
        labels_present = np.unique(i)
        print(f'{path_to_label.stem} : {labels_present}')
        if max(labels_present) > 59:
            warnings.warn(f'invalid label found {labels_present}')
            warning += 1
            warning_msg.append(f'invalid label found {labels_present}')
    return warning_msg, warning

def class_dict_from_txt():
    d = dict()
    content = open('pascal.txt').read()
    print('{')
    for i in content.split('\n'):
        key = i.split(':')[0]
        val = i.split(':')[-1]
        # print(key, val)
        d[int(key)] = val
        val = val.replace(" ", "")
        print(f'{key}:"{val}",')
    print('}')
if __name__ == '__main__':
    # label_sanity_check()
    # class_dict_from_txt()

    # for i in classes_exp0:
    #     # for remapping
    #     # print(f'{i-1}:{[i]},')
    from utils import get_pascalc_colormap
    # A = PALETTE
    #     print(f'{i - 1}:"{classes_exp0[i]}",')
    for i, c in enumerate(classes_exp0):
        print(f'{i-1}:"{c}",')