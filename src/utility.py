

import os
import shutil
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import yaml


class LRFactor:
    def __init__(self, decay, gamma):
        assert len(decay) == len(gamma)

        self.decay = decay
        self.gamma = gamma

    def get_factor(self, epoch):
        for (d, g) in zip(self.decay, self.gamma):
            if epoch < d:
                return g
        return self.gamma[-1]


def convert_str_to_num(val, t):
    val = val.replace('\'', '')
    val = val.replace('\"', '')

    if t == 'int':
        val = [int(v) for v in val.split(',')]
    elif t == 'float':
        val = [float(v) for v in val.split(',')]
    else:
        raise NotImplementedError

    return val


def make_optimizer_scheduler(args, target):
    # optimizer
    if hasattr(target, 'param_groups'):
        # NOTE : lr for each group must be set by the network
        trainable = target.param_groups
    else:
        trainable = filter(lambda x: x.requires_grad, target.parameters())

    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSProp
        kwargs_optimizer['eps'] = args.epsilon
    else:
        raise NotImplementedError

    optimizer = optimizer_class(trainable, **kwargs_optimizer)

    # scheduler
    decay = convert_str_to_num(args.decay, 'int')
    gamma = convert_str_to_num(args.gamma, 'float')

    assert len(decay) == len(gamma), 'decay and gamma must have same length'

    calculator = LRFactor(decay, gamma)
    scheduler = lrs.LambdaLR(optimizer, calculator.get_factor)

    return optimizer, scheduler


def make_optimizer_scheduler_split(args, target):
    # optimizer
    if hasattr(target, 'param_groups'):
        # NOTE : lr for each group must be set by the network
        trainable = target.param_groups
    else:
        # print('split backbone learning rate to 0.01 times of base learning rate')
        backbone_params_list = list(map(id, target.depth_backbone.parameters()))
        base_params = filter(lambda p: id(p) not in backbone_params_list and p.requires_grad, target.parameters())
        backbone_params = filter(lambda x: x.requires_grad, target.depth_backbone.parameters())
        trainable = [
            {'params': base_params},
            {'params': backbone_params, 'lr': 0.1 * args.lr}
        ]

        # trainable = filter(lambda x: x.requires_grad, target.parameters())

    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSProp
        kwargs_optimizer['eps'] = args.epsilon
    else:
        raise NotImplementedError

    optimizer = optimizer_class(trainable, **kwargs_optimizer)

    # scheduler
    decay = convert_str_to_num(args.decay, 'int')
    gamma = convert_str_to_num(args.gamma, 'float')

    assert len(decay) == len(gamma), 'decay and gamma must have same length'

    calculator = LRFactor(decay, gamma)
    scheduler = lrs.LambdaLR(optimizer, calculator.get_factor)

    return optimizer, scheduler


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))


# Minknet
def load_cfg_from_cfg_file(file):
    '''Load from config files.'''

    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, value in cfg_from_file[key].items():
            cfg[k] = value

    cfg = CfgNode(cfg)
    return cfg


def merge_cfg_from_list(cfg, cfg_list):
    '''Merge configs from a list.'''

    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg

