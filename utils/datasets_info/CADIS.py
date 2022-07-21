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


DATA_SPLITS = [  # Pre-defined splits of the videos, to be used generally
    [[1], [5]],  # Split 0: debugging
    [[1, 3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25], [5, 7, 16, 2, 12, 22]],  # Split 1
    [list(range(1, 26)), [5, 7, 16, 2, 12, 22]],  # Split 2 (all data)
    [[1, 8, 9, 10, 14, 15, 21, 23, 24], [5, 7, 16, 2, 12, 22]],     # Split 3: "50% of data" (1729 frames, 49.3%)
    [[10, 14, 21, 24], [5, 7, 16, 2, 12, 22]],                      # Split 4: "25% of data" (834 frames, 23.8%)
    [[1, 3, 4, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25], [5, 7, 16], [2, 12, 22]]  # train-val-test
]

categories_exp0 = {
    'anatomies': [],
    'instruments': [],
    'others': []
}
categories_exp1 = {
    'anatomies': [0, 4, 5, 6],
    'instruments': [7],
    'others': [1, 2, 3],
    'rare': [2]
}
categories_exp2 = {
    'anatomies': [0, 4, 5, 6],
    'instruments': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'others': [1, 2, 3],
    'rare': [16, 10, 9, 12, 14]  # picked with freq_thresh 0.2 and s.t rf > 1.5
}
categories_exp3 = {
    'anatomies': [0, 4, 5, 6],
    'instruments': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
    'others': [1, 2, 3],
    'rare': [24, 20, 21, 22, 18, 23, 19, 16, 12, 11, 14]  # picked with freq_thresh 0.2 and s.t rf > 1.5
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
    34: [34],
    35: [35]
}
classes_exp0 = {
    0: 'Pupil',
    1: 'Surgical Tape',
    2: 'Hand',
    3: 'Eye Retractors',
    4: 'Iris',
    5: 'Skin',
    6: 'Cornea',
    7: 'Hydrodissection Cannula',
    8: 'Viscoelastic Cannula',
    9: 'Capsulorhexis Cystotome',
    10: 'Rycroft Cannula',
    11: 'Bonn Forceps',
    12: 'Primary Knife',
    13: 'Phacoemulsifier Handpiece',
    14: 'Lens Injector',
    15: 'I/A Handpiece',
    16: 'Secondary Knife',
    17: 'Micromanipulator',
    18: 'I/A Handpiece Handle',
    19: 'Capsulorhexis Forceps',
    20: 'Rycroft Cannula Handle',
    21: 'Phacoemulsifier Handpiece Handle',
    22: 'Capsulorhexis Cystotome Handle',
    23: 'Secondary Knife Handle',
    24: 'Lens Injector Handle',
    25: 'Suture Needle',
    26: 'Needle Holder',
    27: 'Charleux Cannula',
    28: 'Primary Knife Handle',
    29: 'Vitrectomy Handpiece',
    30: 'Mendez Ring',
    31: 'Marker',
    32: 'Hydrodissection Cannula Handle',
    33: 'Troutman Forceps',
    34: 'Cotton',
    35: 'Iris Hooks'
}

class_remapping_exp1 = {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
        4: [4],
        5: [5],
        6: [6],
        7: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
    }
classes_exp1 = {
    0: "Pupil",
    1: "Surgical Tape",
    2: "Hand",
    3: "Eye Retractors",
    4: "Iris",
    5: "Skin",
    6: "Cornea",
    7: "Instrument",
}

class_remapping_exp2 = {
    0: [0],
    1: [1],
    2: [2],
    3: [3],
    4: [4],
    5: [5],
    6: [6],
    7: [7, 8, 10, 27, 20, 32],
    8: [9, 22],
    9: [11, 33],
    10: [12, 28],
    11: [13, 21],
    12: [14, 24],
    13: [15, 18],
    14: [16, 23],
    15: [17],
    16: [19],
    255: [25, 26, 29, 30, 31, 34, 35],
}
classes_exp2 = {
    0: "Pupil",
    1: "Surgical Tape",
    2: "Hand",
    3: "Eye Retractors",
    4: "Iris",
    5: "Skin",
    6: "Cornea",
    7: "Cannula",
    8: "Cap. Cystotome",
    9: "Tissue Forceps",
    10: "Primary Knife",
    11: "Ph. Handpiece",
    12: "Lens Injector",
    13: "I/A Handpiece",
    14: "Secondary Knife",
    15: "Micromanipulator",
    16: "Cap. Forceps",
    255: "Ignore",
}

class_remapping_exp3 = {
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
    255: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
}
classes_exp3 = {
    0: "Pupil",
    1: "Surgical Tape",
    2: "Hand",
    3: "Eye Retractors",
    4: "Iris",
    5: "Skin",
    6: "Cornea",
    7: "Hydro. Cannula",
    8: "Visc. Cannula",
    9: "Cap. Cystotome",
    10: "Rycroft Cannula",
    11: "Bonn Forceps",
    12: "Primary Knife",
    13: "Ph. Handpiece",
    14: "Lens Injector",
    15: "I/A Handpiece",
    16: "Secondary Knife",
    17: "Micromanipulator",
    18: "I/A Handpiece Handle",
    19: "Cap. Forceps",
    20: "R. Cannula Handle",
    21: "Ph. Handpiece Handle",
    22: "Cap. Cystotome Handle",
    23: "Sec. Knife Handle",
    24: "Lens Injector Handle",
    255: "Ignore",
}

CLASS_INFO = [
    [class_remapping_exp0, classes_exp0, categories_exp0],  # Original classes
    [class_remapping_exp1, classes_exp1, categories_exp1],
    [class_remapping_exp2, classes_exp2, categories_exp2],
    [class_remapping_exp3, classes_exp3, categories_exp3]
]

CLASS_NAMES = [[CLASS_INFO[0][1][key] for key in sorted(CLASS_INFO[0][1].keys())],
               [CLASS_INFO[1][1][key] for key in sorted(CLASS_INFO[1][1].keys())],
               [CLASS_INFO[2][1][key] for key in sorted(CLASS_INFO[2][1].keys())],
               [CLASS_INFO[3][1][key] for key in sorted(CLASS_INFO[3][1].keys())]]

OVERSAMPLING_PRESETS = {
    'default': [
        [3, 5, 7],            # Experiment 1
        [7, 8, 15, 16],       # Experiment 2
        [19, 20, 22, 24]      # Experiment 3
    ],
    'rare': [  # Same classes as 'rare' category for mIoU metric
        [2],                                            # Experiment 1
        [16, 10, 9, 12, 14],                            # Experiment 2
        [24, 20, 21, 22, 18, 23, 19, 16, 12, 11, 14]    # Experiment 3
    ]
}

CLASS_FREQUENCIES = [
    1.68024535e-01,
    5.93061223e-02,
    7.38987570e-03,
    5.72173439e-03,
    1.12288211e-01,
    1.33608027e-01,
    4.89257831e-01,
    1.26300163e-03,
    8.96526043e-04,
    9.28408858e-04,
    6.47719387e-04,
    2.61340734e-03,
    1.40455685e-03,
    1.84766048e-03,
    3.25327478e-03,
    3.60986861e-03,
    1.06050077e-03,
    1.97264561e-03,
    5.32642854e-04,
    7.07037962e-04,
    3.66272768e-04,
    4.75095501e-04,
    1.73250919e-04,
    5.49602466e-04,
    2.91966965e-04,
    1.06066764e-05,
    1.54437472e-04,
    4.16546878e-05,
    2.96828324e-06,
    1.02785378e-04,
    4.38665256e-04,
    4.91079867e-04,
    1.13576281e-05,
    1.83788200e-04,
    1.37330396e-04,
    2.35550169e-04
]
CLASS_SUMS = [
    406775301,
    143575852,
    17890357,
    13851907,
    271841675,
    323455413,
    1184457982,
    3057636,
    2170425,
    2247611,
    1568082,
    6326871,
    3400331,
    4473053,
    7875944,
    8739232,
    2567396,
    4775633,
    1289490,
    1711688,
    886720,
    1150172,
    419428,
    1330548,
    706831,
    25678,
    373882,
    100843,
    7186,
    248836,
    1061977,
    1188869,
    27496,
    444938,
    332467,
    570250
]

CADIS_INFO = EasyDict(CLASS_INFO=CLASS_INFO,
                      CLASS_NAMES=CLASS_NAMES,
                      DATA_SPLITS=DATA_SPLITS,
                      OVERSAMPLING_PRESETS=OVERSAMPLING_PRESETS,
                      CLASS_FREQUENCIES=CLASS_FREQUENCIES,
                      CLASS_SUMS=CLASS_SUMS)
