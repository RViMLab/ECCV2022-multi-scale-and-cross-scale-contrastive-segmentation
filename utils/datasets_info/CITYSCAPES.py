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
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33]
}

categories_exp1 = {
    'flat': [0, 1],
    'construction': [2, 3, 4],
    'object': [5, 6, 7],
    'nature': [8, 9],
    'sky': [10],
    'human': [11, 12],
    'vehicle': [13, 14, 15, 16, 17, 18]
}

class_remapping_exp0 = {
    0: [0],
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
    -1: [-1]
}
classes_exp0 = {
    0: 'unlabeled',
    1: 'ego vehicle',
    2: 'rectification border',
    3: 'out of roi',
    4: 'static',
    5: 'dynamic',
    6: 'ground',
    7: 'road',
    8: 'sidewalk',
    9: 'parking',
    10: 'rail track',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
    21: 'vegetation',
    22: 'terrain',
    23: 'sky',
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle',
    34: 'Cotton',
    35: 'Iris Hooks',
    -1: 'license plate'
}

class_remapping_exp1 = {
    0: [7],
    1: [8],
    2: [11],
    3: [12],
    4: [13],
    5: [17],
    6: [19],
    7: [20],
    8: [21],
    9: [22],
    10: [23],
    11: [24],
    12: [25],
    13: [26],
    14: [27],
    15: [28],
    16: [31],
    17: [32],
    18: [33],
    255: [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
}


classes_exp1 = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
    255: 'Ignore'
}


CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1]
]

CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())]]

CITYSCAPES_INFO = EasyDict(CLASS_INFO=CLASS_INFO, CLASS_NAMES=CLASS_NAMES)

if __name__ == '__main__':
    # all info is in a class attribute of the Cityscapes class
    from torchvision.datasets.cityscapes import Cityscapes
    CTS_info = Cityscapes.classes
    categories_exp1 = {}
    colormap = {}
    ingored_colormap = {}
    class_remap_exp1 = {}
    categ_exp0 = {}
    categ_exp1 = {}
    for cl in CTS_info:
        ############################################
        classes_exp0[cl.id] = cl.name
        colormap[cl.id] = cl.color
        # ingored_colormap[cl.train_id] = cl.color
        ############################################
        if cl.train_id in class_remap_exp1:
            class_remap_exp1[cl.train_id] += [cl.id]
        else:
            # -1 mapped to 255 which is used as the ignored class
            class_remap_exp1[cl.train_id] = [cl.id]
        classes_exp1[cl.train_id] = cl.name

        if cl.category not in categ_exp0:
            categ_exp0[cl.category] = [cl.id]
        else:
            categ_exp0[cl.category] += [cl.id]

        if cl.category not in categ_exp1:
            categ_exp1[cl.category] = [cl.train_id]
        else:
            categ_exp1[cl.category] += [cl.train_id]

    class_remap_exp1.pop(-1)  # remove -1 from dictionary
    class_remap_exp1[255] += [-1]  # and place it in the ignore class

    classes_exp1[255] = 'Ignore'
    classes_exp1.pop(-1)  # remove -1 from dictionary

    a = 1
