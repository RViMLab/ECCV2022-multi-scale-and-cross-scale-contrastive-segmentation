import pathlib
import json
from .defaults import DEFAULT_CONFIG_DICT, DEFAULT_CONFIG_NESTED_DICT, DATASETS_INFO
from .logger import printlog
from torchvision.transforms import ToPILImage, ColorJitter, ToTensor, Normalize, RandomApply
from .transforms import BlurPIL, RandomCropImgLbl, RandomResize, Resize
from .np_transforms import AffineNP, PadNP, FlipNP


def parse_config(file_path, user, device, dataset, parallel):
    # Load config
    try:
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
    except FileNotFoundError:
        print("Configuration file not found at given path '{}'".format(file_path))
        exit(1)
    # Fill in correct paths
    config_path = pathlib.Path(file_path).parent.parent
    with open(config_path / 'path_info.json', 'r') as f:
        path_info = json.load(f)  # Dict: keys are user codes, values are a list of 'data_path', 'log_path' (absolute)
    if dataset != -1:  # if dataset provided as an argument for main
        assert(dataset in ['CITYSCAPES', 'CADIS', 'PASCALC', 'ADE20K'])
        config_dict['data']['dataset'] = dataset
        # print('dataset set to {} '.format(dataset))
    else:
        dataset = config_dict['data']['dataset']

    dataset_user_suffix = ''
    if dataset == 'CITYSCAPES':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_CTS'
        config_dict['parallel'] = parallel
    if dataset == 'PASCALC':
        config_dict['data']['experiment'] = 1
        dataset_user_suffix = '_PASCALC'
        config_dict['parallel'] = parallel
    if dataset == 'CADIS':
        dataset_user_suffix = '_CADIS'
        config_dict['parallel'] = parallel
    if dataset == 'ADE20K':
        dataset_user_suffix = '_ADE20K'
        config_dict['data']['experiment'] = 1
        config_dict['parallel'] = parallel
    if dataset == 'IACL':
        dataset_user_suffix = '_IACL'
        config_dict['data']['experiment'] = 1
        config_dict['parallel'] = parallel
    if dataset == 'RETOUCH':
        dataset_user_suffix = '_RETOUCH'
        config_dict['data']['experiment'] = 1
        config_dict['parallel'] = parallel

    if user+dataset_user_suffix in path_info:
        config_dict.update({
            'data_path': pathlib.Path(path_info[user+dataset_user_suffix][0]),
            'log_path': pathlib.Path(path_info[user+dataset_user_suffix][1])
        })
    else:
        ValueError("User '{}' not found in configs/path_info.json".format(user))
    assert config_dict['data_path'].exists(), 'data_path {} does not exist'.format(config_dict['data_path'])
    assert config_dict['log_path'].exists(),  'log_path  {} does not exist'.format(config_dict['log_path'])
    # Fill in GPU device if applicable

    if isinstance(device, list):
        config_dict['gpu_device'] = device
    elif device >= 0:  # Only update config if user entered a device (default otherwise -1)
        config_dict['gpu_device'] = device

    # Make sure all ne0000cessary default values exist
    default_dict = DEFAULT_CONFIG_DICT.copy()
    default_dict.update(config_dict)  # Keeps all default values not overwritten by the passed config
    nested_default_dicts = DEFAULT_CONFIG_NESTED_DICT.copy()
    for k, v in nested_default_dicts.items():  # Go through the nested dicts, set as default first, then update
        default_dict[k] = v  # reset to default values
        default_dict[k].update(config_dict[k])  # Overwrite defaults with the passed config values

    # Extra config bits needed
    default_dict['data']['transform_values']['experiment'] = default_dict['data']['experiment']
    return default_dict


def parse_transform_list_legacy(transform_list, transform_values, num_classes, dataset, experiment):
    """Helper function to parse given dataset transform list. Order of things:
    - first the 'common' transforms are applied. At this point, the input is expected to be a numpy array.
    - then the img and lbl transforms are each applied as necessary. The input is expected to be a numpy array, the
        output will be a tensor, as required by PyTorch"""
    d = {"dataset": dataset, "experiment": experiment}

    printlog("------parsing transform list------")
    transforms_dict = {
        'train': {
            'common': [],
            'img': [],
            'lbl': [],
        },
        'valid': {
            'common': [],
            'img': [],
            'lbl': [],
        }
    }

    # Step 1: Go through all transforms that need to go into the 'commom' section, i.e. which rely on using the same
    # random parameters on both the image and the label: generally actual augmentation transforms.
    #   Input: np.ndarray; Output: np.ndarray
    if 'flip' in transform_list:
        transforms_dict['train']['common'].append(FlipNP())

    rotation = 0
    rot_centre_offset = (.2, .2)
    shift = 0
    shear = (0, 0)
    shear_centre_offset = (.2, .2)
    set_affine = False
    if 'rot' in transform_list:
        rotation = 15
        set_affine = True
    if 'shift' in transform_list:
        shift = .1
        set_affine = True
    if 'shear' in transform_list:
        shear = (.1, .1)
        set_affine = True
    if 'affine' in transform_list:
        rotation = 10
        shear = (.1, .1)
        rot_centre_offset = (.1, .1)
        set_affine = True

    if set_affine:
        transforms_dict['train']['common'].append(AffineNP(num_classes,
                                                           crop_to_fit=False,
                                                           rotation=rotation,
                                                           rot_centre_offset=rot_centre_offset,
                                                           shift=shift,
                                                           shear=shear,
                                                           shear_centre_offset=shear_centre_offset))


    # Step 2: Go through all transforms that need to be applied individually afterwards
    #
    # Pad (if necessary) will be the first element of 'img' / 'lbl' transforms.
    #   Input: np.ndarray; Output: np.ndarray
    if 'pad' in transform_list:
        # Needs to be added to img and lbl, train and valid
        if 'crop' not in transform_list:  # Padding only necessary if no cropping has happened
            for obj in ['img', 'lbl']:
                transforms_dict['train'][obj].append(PadNP(ver=(2, 2), hor=(0, 0), padding_mode='reflect'))
        for obj in ['img', 'lbl']:  # Padding for validation always necessary, as never cropped
            transforms_dict['valid'][obj].append(PadNP(ver=(2, 2), hor=(0, 0), padding_mode='reflect'))

    # PIL Image: needed for training images if some of the pytorch transform functions are present
    pil_needed = False
    for t in transform_list:
        if t in ['colorjitter', 'blur', 'crop2', 'random_scale' ]:  # Add other keywords for fcts that need pil.Image input here
            pil_needed = True
    if pil_needed:
        transforms_dict['train']['img'].append(ToPILImage())

    if 'resize' in transform_list:
        transforms_dict['train']['common'].append(Resize(min_side_length=transform_values['min_side_length'],
                                                         fit_stride=None))

    # ColorJitter only applied on training images
    #   Input: pil.Image; Output: pil.Image
    if 'random_scale' in transform_list:
        transforms_dict['train']['common'].append(RandomResize(**d,
                                                               scale_range=transform_values['scale_range'],
                                                               target_size=transform_values['crop_shape'],
                                                               aspect_range=[0.9, 1.1], probability=1.0))

    if 'crop2' in transform_list:
        max_ratio = transform_values['crop_class_max_ratio'] if 'crop_class_max_ratio' in transform_values else None

        transforms_dict['train']['common'].append(RandomCropImgLbl(**d,
                                                                   shape=transform_values['crop_shape'],
                                                                   crop_class_max_ratio=max_ratio))

    if 'blur' in transform_list:
        transforms_dict['train']['img'].append(BlurPIL(**d, probability=.05, kernel_limits=(3, 7)))

    if 'colorjitter' in transform_list:
        transforms_dict['train']['img'].append(ColorJitter(brightness=(2/3, 1.5), contrast=(2/3, 1.5),
                                                           saturation=(2/3, 1.5), hue=(-.05, .05)))

    s = None
    # for using stronger than default augmentation -- may be removed in the future
    if 'pseudo_colorjitter' in transform_list:
        for e in transform_list:
            if isinstance(e, dict):
                if 'colorjitter_strength' in e:
                    s = e['colorjitter_strength']
                    assert s in [1, 2, 3]
                    break
        if s is None:
            s = 2
        range_extent = (1-s*0.25, 1+s*0.25)
        color_jitter = ColorJitter(brightness=range_extent,
                                   contrast=range_extent,
                                   saturation=range_extent,
                                   hue=(-.02 * s, .02 * s))
        rnd_color_jitter = RandomApply([color_jitter], p=0.7)
        transforms_dict['train']['img'].append(rnd_color_jitter)

    if 'resize_val' in transform_list:
        # for Pascal Context: resizes with min_side, aspect ratio preserved,
        # pad to fit_Stride and returns original labels for validation
        # transforms_dict['valid']['img'].append(Pad(min_side_length=transform_values['min_side_length']))
        transforms_dict['valid']['common'].append(Resize(**d,
                                                         min_side_length=transform_values['min_side_length'],
                                                         fit_stride=transform_values['fit_stride_val'],
                                                         return_original_labels=True))

    # Tensor: needed by default.
    #   Input: np.array or pil.Image; Output: torch.Tensor
    for stage in ['train', 'valid']:
        for obj in ['img', 'lbl']:
            transforms_dict[stage][obj].append(ToTensor())

    # Normalisation (e.g. for use with the pretrained ResNets from the torchvision model zoo)
    #   Input: torch.Tensor; Output: torch.Tensor
    if 'torchvision_normalise' in transform_list:
        for stage in ['train', 'valid']:
            # imagenet means/std subtract
            transforms_dict[stage]['img'].append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    printlog("----------------------------------")
    return transforms_dict


def parse_transform_lists(transform_list, transform_values, dataset, experiment):
    """Helper function to parse given dataset transform list. Order of things:
    - first the 'common' transforms are applied. At this point, the input is expected to be a numpy array.
    - then the img and lbl transforms are each applied as necessary. The input is expected to be a numpy array, the
        output will be a tensor, as required by PyTorch"""
    d = {"dataset": dataset, "experiment": experiment}
    num_classes = len(DATASETS_INFO[dataset].CLASS_INFO[experiment][1]) - 1 if 255 in DATASETS_INFO[dataset].CLASS_INFO[experiment][1].keys() \
        else len(DATASETS_INFO[dataset].CLASS_INFO[experiment][1])

    printlog("------parsing transform list {}------")
    transforms_dict = \
        {
            'common': [],
            'img': [],
            'lbl': []
        }

    # Step 1: Go through all transforms that need to go into the 'commom' section, i.e. which rely on using the same
    # random parameters on both the image and the label: generally actual augmentation transforms.
    #   Input: np.ndarray/PIL; Output: np.ndarray
    first = True
    for t in transform_list:
        if t == 'flip':
            transforms_dict['common'].append(FlipNP())
            # if first:
            #     transforms_dict = determine_affine(transform_list, transforms_dict, num_classes)
            #     first = False
        elif t == 'pad':
            assert dataset=='CADIS'
            # this is for CADIS only
            # Needs to be added to img and lbl, train and valid
            if 'crop' not in transform_list:  # Padding only necessary if no cropping has happened
                for obj in ['img', 'lbl']:
                    transforms_dict[obj].append(PadNP(ver=(2, 2), hor=(0, 0), padding_mode='reflect'))

        elif t == 'resize':
            fit_stride = transform_values['fit_stride'] if 'fit_stride' in transform_values else None
            target_size = transform_values['target_size'] if 'target_size' in transform_values else None
            min_side_length = transform_values['min_side_length'] if 'min_size_length' in transform_values else None
            transforms_dict['common'].append(Resize(**d,
                                                    target_size=target_size,
                                                    min_side_length=min_side_length,
                                                    fit_stride=fit_stride))
        elif t == 'resize_val':
            # e.x Pascal Context: resizes with min_side, aspect ratio preserved,
            # pad to fit_Stride and returns original labels for validation
            transforms_dict['common'].append(Resize(**d,
                                                    min_side_length=transform_values['min_side_length'],
                                                    fit_stride=transform_values['fit_stride_val'],
                                                    return_original_labels=True))
        elif t == 'random_scale':
            aspect_range = transform_values['aspect_range'] if 'aspect_range' in transform_values else [0.9, 1.1]
            p_random_scale = transform_values['p_random_scale'] if 'p_random_scale' in transform_values else 1.0
            transforms_dict['common'].append(RandomResize(**d,
                                                          scale_range=transform_values['scale_range'],
                                                          target_size=transform_values['crop_shape'],
                                                          aspect_range=aspect_range,
                                                          probability=p_random_scale))

        elif t == 'RandomCropImgLbl':
            max_ratio = transform_values['crop_class_max_ratio'] if 'crop_class_max_ratio' in transform_values else None
            transforms_dict['common'].append(RandomCropImgLbl(**d,
                                                              shape=transform_values['crop_shape'],
                                                              crop_class_max_ratio=max_ratio))

        elif t == 'blur':
            transforms_dict['img'].append(BlurPIL(**d,
                                                  probability=.05,
                                                  kernel_limits=(3, 7)))

        elif t == 'colorjitter':

            transforms_dict['img'].append(ToPILImage())
            transforms_dict['lbl'].append(ToPILImage())

            p = transform_values['colorjitter_p'] if 'colorjitter_p' in transform_values else 1.0

            colorjitter_func = ColorJitter(brightness=(2 / 3, 1.5),
                                            contrast=(2 / 3, 1.5),
                                            saturation=(2 / 3, 1.5),
                                            hue=(-.05, .05))
            rnd_color_jitter = colorjitter_func
            # rnd_color_jitter =  RandomApply([colorjitter_func], p=p)
            transforms_dict['img'].append(rnd_color_jitter)
            printlog(f'{colorjitter_func} with p {p}')

        elif t == 'pseudo_colorjitter':
            # for using stronger/weaker than default colorjitter -- may be removed in the future
            s = transform_values['colorjitter_strength'] if 'colorjitter_strength' in transform_values else 2
            p = transform_values['p_colorjitter'] if 'p_colorjitter' in transform_values else 0.7
            range_extent = (1 - s * 0.25, 1 + s * 0.25)
            color_jitter = ColorJitter(brightness=range_extent,
                                       contrast=range_extent,
                                       saturation=range_extent,
                                       hue=(-.02 * s, .02 * s))
            rnd_color_jitter = RandomApply([color_jitter], p=p)
            transforms_dict['img'].append(rnd_color_jitter)
        elif t in ['torchvision_normalise']:
            a = 1
        else:
            raise ValueError(f'transform {t} not recognized')

    for obj in ['img', 'lbl']:
        transforms_dict[obj].append(ToTensor())
    if 'torchvision_normalise' in transform_list:
                transforms_dict['img'].append(
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    printlog("----------------------------------")
    return transforms_dict



def determine_affine(transform_list, transforms_dict, num_classes):
    rotation = 0
    rot_centre_offset = (.2, .2)
    shift = 0
    shear = (0, 0)
    shear_centre_offset = (.2, .2)
    set_affine = False
    if 'rot' in transform_list:
        rotation = 15
        set_affine = True
    if 'shift' in transform_list:
        shift = .1
        set_affine = True
    if 'shear' in transform_list:
        shear = (.1, .1)
        set_affine = True
    if 'affine' in transform_list:
        rotation = 10
        shear = (.1, .1)
        rot_centre_offset = (.1, .1)
        set_affine = True

    if set_affine:
        transforms_dict['train']['common'].append(AffineNP(num_classes=num_classes,
                                                           crop_to_fit=False,
                                                           rotation=rotation,
                                                           rot_centre_offset=rot_centre_offset,
                                                           shift=shift,
                                                           shear=shear,
                                                           shear_centre_offset=shear_centre_offset))


    return transforms_dict