# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-20

import torch
from torch import nn
from opennlp.encoder.textrcnn import TextRCNN
from opennlp.encoder.textrnn import TextRNN


TextRCNN, TextRNN


class EncoderType:
    TextRCNN = "TextRCNN"
    TextRNN = "TextRNN"



class EncodeLayer(nn.Module):
    def __init__(self, config):
        super(EncodeLayer, self).__init__()
        self.config = config
        self.encoder = globals()[config.task.model_name](config)

    def get_hidden_size(self):
        return self.encoder.hidden_size

    def get_optimize_parameters(self):
        return self.encoder.get_optimize_parameters()

    def forward(self, embedding, **kwargs):
        return self.encoder(embedding, **kwargs)
