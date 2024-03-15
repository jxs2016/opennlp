# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-16

from torch import nn
from opennlp.decoder.linear import Linear


class DecodeLayer(nn.Module):
    def __init__(self, config, hidden_size, label_size):
        super(DecodeLayer, self).__init__()
        self.decoder = Linear(hidden_size, label_size)
        self.dropout = nn.Dropout(p=config.train.hidden_dropout)

    def get_optimize_parameters(self):
        return self.decoder.get_optimize_parameters()

    def forward(self, embedding, *args, **kwargs):
        embedding = self.decoder(embedding, *args, **kwargs)
        return self.dropout(embedding)