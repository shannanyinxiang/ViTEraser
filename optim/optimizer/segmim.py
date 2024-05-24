import torch.optim as optim

from functools import partial

def build_optimizer(model, args):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    if args.segmim_finetune:
        num_layers = sum(args.swin_enc_depths)
        get_layer_func = partial(get_swin_layer, num_layers=num_layers + 2, depths=args.swin_enc_depths)
        scales = list(args.layer_decay ** i for i in reversed(range(num_layers + 2)))
        parameters = get_finetune_param_groups(model, args.lr, 0.05, get_layer_func, scales, skip, skip_keywords)
    else:
        parameters = get_pretrain_param_groups(model, skip, skip_keywords)
    optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999), lr=args.lr, weight_decay=0.05)

    return optimizer

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def get_swin_layer(name, num_layers, depths):
    name = name.replace('module.encoder.backbone.', '')
    if name in ("mask_token"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("layers"):
        layer_id = int(name.split('.')[1])
        block_id = name.split('.')[3]
        if block_id == 'reduction' or block_id == 'norm':
            return sum(depths[:layer_id + 1])
        layer_id = sum(depths[:layer_id]) + int(block_id)
        return layer_id + 1
    else:
        return num_layers - 1


def get_finetune_param_groups(model, lr, weight_decay, get_layer_func, scales, skip_list=(), skip_keywords=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_layer_func is not None:
            layer_id = get_layer_func(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if scales is not None:
                scale = scales[layer_id]
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale,
            }
            parameter_group_vars[group_name] = {
                "group_name": group_name,
                "weight_decay": this_weight_decay,
                "params": [],
                "lr": lr * scale,
                "lr_scale": scale
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values())