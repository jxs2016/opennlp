# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-27

from enum import Enum
import torch


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class OptimizerType:

    Adam = "Adam"
    AdamW = "AdamW"
    Adamax = "Adamax"
    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    SparseAdam = "SparseAdam"

    @classmethod
    def str(cls):
        return ",".join([cls.Adam, cls.AdamW, cls.Adamax, cls.Adadelta, cls.SparseAdam, cls.Adagrad])


class Optimizer:
    def __init__(self, config, model):
        self.config = config
        self.model = model

        if hasattr(self.model, "module"):
            self.params = self.model.module.get_optimize_parameters()
        else:
            self.params = self.model.get_optimize_parameters()

    def get_optimizer(self):
        if self.config.optimize_type == OptimizerType.Adam:
            return torch.optim.Adam(self.params,
                                    lr=self.config.learning_rate,
                                    betas=(self.config.beta1, self.config.beta2),
                                    eps=self.config.epsilon,
                                    weight_decay=self.config.weight_decay,
                                    amsgrad=self.config.amsgrad)
        elif self.config.optimize_type == OptimizerType.Adadelta:
            return torch.optim.Adadelta(self.params,
                                        lr=self.config.learning_rate,
                                        rho=self.config.beta1,
                                        eps=self.config.epsilon,
                                        weight_decay=self.config.weight_decay)
        elif self.config.optimize_type == OptimizerType.Adagrad:
            return torch.optim.Adagrad(self.params,
                                       lr=self.config.learning_rate,
                                       eps=self.config.epsilon,
                                       weight_decay=self.config.weight_decay,
                                       lr_decay=0, initial_accumulator_value=0)
        elif self.config.optimize_type == OptimizerType.AdamW:
            return torch.optim.AdamW(self.params,
                                     lr=self.config.learning_rate,
                                     betas=(self.config.beta1, self.config.beta2),
                                     eps=self.config.epsilon,
                                     weight_decay=self.config.weight_decay,
                                     amsgrad=self.config.amsgrad)
        elif self.config.optimize_type == OptimizerType.Adamax:
            return torch.optim.Adamax(self.params,
                                      lr=self.config.learning_rate,
                                      betas=(self.config.beta1, self.config.beta2),
                                      eps=self.config.epsilon,
                                      weight_decay=self.config.weight_decay)
        elif self.config.optimize_type == OptimizerType.Adamax:
            return torch.optim.SparseAdam(self.params,
                                          lr=self.config.learning_rate,
                                          betas=(self.config.beta1, self.config.beta2),
                                          eps=self.config.epsilon)

        else:
            raise TypeError(
                "Unsupported tensor optimizer type: %s.Supported optimizer "
                "type is: %s" % (self.config.optimize_type, OptimizerType.str()))

