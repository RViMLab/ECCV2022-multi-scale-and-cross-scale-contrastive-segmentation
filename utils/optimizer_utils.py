from .logger import printlog

def get_num_layer_stage_wise(var_name, num_max_layer):
    """Get the layer id to set the different learning rates in ``stage_wise``
    decay_type. only for convnext series
    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.
    Returns:
        int: The id number corresponding to different　learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token', 'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return num_max_layer - 1 # this essentially means all layers beyond layers get lr = base_lr


def is_in(param_group, param_group_list):
    # assert is_list_of(param_group_list, dict)
    param = set(param_group['params'])
    param_set = set()
    for group in param_group_list:
        param_set.update(set(group['params']))
    return not param.isdisjoint(param_set)


def get_param_groups_using_keys(model, config):
    # this mode specifies keys (strings) that if present in var's name place the var in a parameter group
    # each such key generates a new param_group where all variables share wd_mult, lr_mult
    base_lr = config['train']['learning_rate']
    base_wd = config['train']['weight_decay']
    params = []
    parameter_groups = {}
    params_dict = dict(model.named_parameters())
    for name, param in params_dict.items():
        param_group = {'params': [param], "group_name": name, 'param_names': []}
        is_custom = (False, None)
        is_first_in_group = False
        if not param.requires_grad:
            params.append(param_group)
            continue
        if is_in(param_group, params):
            a = 1
        group_name = 'base_lr_wd'
        for custom_key in config['train']['opt_keys']:
            if custom_key in name:
                is_custom = (True, custom_key)
                lr_mult = config['train']['opt_keys'][custom_key].get('lr_mult', 1.0)
                wd_mult = config['train']['opt_keys'][custom_key].get('wd_mult', 1.0)
                param_group['lr'] = lr_mult * base_lr
                param_group['weight_decay'] = wd_mult * base_wd
                group_name = f'{custom_key}_lrm{lr_mult}_wdm{wd_mult}'
                break
        if not is_custom[0]:
            param_group['lr'] = base_lr
            param_group['weight_decay'] = base_wd

        if group_name not in parameter_groups:
            param_group['group_name'] = group_name
            parameter_groups[group_name] = param_group
            is_first_in_group = True
            # parameter_groups[group_name]['param_names'] = [name]
        if not is_first_in_group:
            parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)

    params.extend(parameter_groups.values())
    # printlog(f'optimizer param groups : \n {params}')
    params_cnt = 0
    for g in params:
        params_cnt += len(g['param_names'])
    assert (len(params_dict) == params_cnt), f'mismatch between params in parameter groups {params_cnt}' \
                                             f' and model.named_parameters {len(params_dict)}'
    return params

def get_param_groups_with_stage_wise_lr_decay(model, config):
    # adapted from convnext repo
    # scales the learning rate of deeper layers by decay_rate ** (num_layers - layer_id - 1)
    # tl,dr --> latest layers have gradually higher lr
    assert 'ConvNext' in config['graph']['backbone'], f"stage_wise_lr currently only supported for " \
                                                           f"ConvNext backbones instead got {config['graph']['backbone']}"
    decay_rate = config['train']['stage_wise_lr']['decay_rate']
    num_layers = config['train']['stage_wise_lr']['num_layers'] + 2  # todo +2 is still a mystery (?)
    base_lr = config['train']['learning_rate']
    base_wd = config['train']['weight_decay']
    params = []
    parameter_groups = {}
    params_dict = dict(model.named_parameters())

    for name, param in params_dict.items():
        if len(param.shape) == 1 or name.endswith('.bias') or name in ('pos_embed', 'cls_token'):
            # param.shape == 1 is here to ensure some layer-norm modules have 0 weight decay
            # despite not containing the word "norm" in their names
            # for convnext these are for e.x  'backbone.downsample_layers.0.1.weight'
            # or 'backbone.downsample_layers.0.1.bias'
            group_name = 'no_decay'
            this_weight_decay = 0.0
            # printlog(name, this_weight_decay)
        else:
            group_name = 'decay'
            this_weight_decay = base_wd
        layer_id = get_num_layer_stage_wise(name, num_layers)
        # logger.info(f'set param {name} as id {layer_id}')
        group_name = f'layer_{layer_id}_{group_name}'
        if group_name not in parameter_groups:
            scale = decay_rate ** (num_layers - layer_id - 1) # scale * base_lr is the learning rate for this group
            # printlog(group_name, scale)
            parameter_groups[group_name] = {
                'weight_decay': this_weight_decay,
                'params': [],
                'param_names': [],
                'lr_scale': scale,
                'group_name': group_name,
                'lr': scale * base_lr,
            }
        parameter_groups[group_name]['params'].append(param)
        parameter_groups[group_name]['param_names'].append(name)
    params.extend(parameter_groups.values())
    # printlog(f'optimizer param groups : \n {params}')
    params_cnt = 0
    for g in params:
        params_cnt += len(g['param_names'])
    assert (len(params_dict) == params_cnt), f'mismatch between params in parameter groups {params_cnt}' \
                                             f' and model.named_parameters {len(params_dict)}'
    return params
