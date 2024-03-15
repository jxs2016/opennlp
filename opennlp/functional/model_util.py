# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-22

from torch import nn


class InitType:
    """Standard names for init
    """
    UNIFORM = 'uniform'
    NORMAL = "normal"
    XAVIER_UNIFORM = 'xavier_uniform'
    XAVIER_NORMAL = 'xavier_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'
    KAIMING_NORMAL = 'kaiming_normal'
    ORTHOGONAL = 'orthogonal'

    def str(self):
        return ",".join(
            [self.UNIFORM, self.NORMAL, self.XAVIER_UNIFORM, self.XAVIER_NORMAL,
             self.KAIMING_UNIFORM, self.KAIMING_NORMAL, self.ORTHOGONAL])


class FAN_MODE:
    """Standard names for fan mode
    """
    FAN_IN = 'FAN_IN'
    FAN_OUT = "FAN_OUT"

    def str(self):
        return ",".join([self.FAN_IN, self.FAN_OUT])


class ActivationType:
    """Standard names for activation
    """
    SIGMOID = 'sigmoid'
    TANH = "tanh"
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    NONE = 'linear'

    @classmethod
    def str(cls):
        return ",".join(
            [cls.SIGMOID, cls.TANH, cls.RELU, cls.LEAKY_RELU, cls.NONE])


def init_tensor(tensor, args):
    if args.init_type == InitType.UNIFORM:
        return nn.init.uniform_(tensor, a=-args.uniform_bound, b=args.uniform_bound)
    elif args.init_type == InitType.NORMAL:
        return nn.init.normal_(tensor, mean=args.normal_mean, std=args.normal_std)
    elif args.init_type == InitType.XAVIER_UNIFORM:
        return nn.init.xavier_uniform_(
            tensor, gain=nn.init.calculate_gain(args.activation_type))
    elif args.init_type == InitType.XAVIER_NORMAL:
        return nn.init.xavier_normal_(
            tensor, gain=nn.init.calculate_gain(args.activation_type))
    elif args.init_type == InitType.KAIMING_UNIFORM:
        return nn.init.kaiming_uniform_(
            tensor, a=args.negative_slope, mode=args.kaiming_fan_mode,
            nonlinearity=args.activation_type)
    elif args.init_type == InitType.KAIMING_NORMAL:
        return nn.init.kaiming_normal_(
            tensor, a=args.negative_slope, mode=args.kaiming_fan_mode,
            nonlinearity=args.activation_type)
    elif args.init_type == InitType.ORTHOGONAL:
        return nn.init.orthogonal_(
            tensor, gain=nn.init.calculate_gain(args.activation_type))
    else:
        raise TypeError(
            "Unsupported tensor init type: %s. Supported init type is: %s" % (
                args.init_type, InitType().str()))
