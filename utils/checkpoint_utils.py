from torch import nn


def _check_model_param_prefix(state_dict, prefix:str):
    # check if parameters of model state dict contain a give prefix
    found_prefix_model = False
    for param_name in state_dict:
        if not param_name.startswith(prefix):
            found_prefix_model = False
            if found_prefix_model:
                raise Warning('module prefix found in some of the model params but not others '
                              '-- this will cause bugs!! -- check before proceeding')
            break
        else:
            found_prefix_model = True
    return found_prefix_model


def check_module_prefix(chkpt_state_dict, model:nn.Module):
    found_prefix_model = _check_model_param_prefix(model.state_dict(), prefix='module.')
    found_prefix_chkpt = _check_model_param_prefix(chkpt_state_dict, prefix='module.')

    # remove prefix from chkpt_state_dict keys
    # if that prefix is not found in model variable names
    if ~found_prefix_model and found_prefix_chkpt:
        for k in list(chkpt_state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.'):
                # remove prefix
                chkpt_state_dict[k[len("module."):]] = chkpt_state_dict[k]
            # delete renamed or unused k
            del chkpt_state_dict[k]

    return chkpt_state_dict
