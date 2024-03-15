# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-16

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, hidden_size, label_size):
        super(Linear, self).__init__()
        self.linear = torch.nn.Linear(hidden_size, label_size)

    def get_optimize_parameters(self):
        params = [{'params': self.linear.parameters()}]
        return params

    def forward(self, embedding, *args, **kwargs):
        return self.linear(embedding)