# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-20

import torch
from torch import nn
from opennlp.functional.rnn import RNN


class TextRCNN(nn.Module):
    """TextRNN + TextRNN
    """
    def __init__(self, config):
        super(TextRCNN, self).__init__()
        self.rnn = RNN(
            config.embedding.dimension, config.TextRCNN.hidden_dimension,
            num_layers=config.TextRCNN.num_layers,
            batch_first=True, bidirectional=config.TextRCNN.bidirectional,
            rnn_type=config.TextRCNN.rnn_type)

        hidden_dimension = config.TextRCNN.hidden_dimension
        if config.TextRCNN.bidirectional:
            hidden_dimension *= 2
        self.kernel_sizes = config.TextRCNN.kernel_sizes
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(nn.Conv1d(
                hidden_dimension, config.TextRCNN.num_kernels,
                kernel_size, padding=kernel_size - 1))

        self.top_k = config.TextRCNN.top_k_max_pooling
        self.hidden_size = len(config.TextRCNN.kernel_sizes) * \
                      config.TextRCNN.num_kernels * self.top_k

    def get_optimize_parameters(self):
        params = list()
        params.append({'params': self.rnn.parameters()})
        params.append({'params': self.convs.parameters()})
        return params

    def forward(self, embedding, **kwargs):
        input_lengths = kwargs.get("input_lengths")
        assert input_lengths is not None, "input_lengths should not be None."
        output, last_hidden = self.rnn(embedding, input_lengths)
        doc_embedding = output.transpose(1, 2)
        pooled_outputs = []
        for _, conv in enumerate(self.convs):
            convolution = nn.functional.relu(conv(doc_embedding))
            pooled = torch.topk(convolution, self.top_k)[0].view(
                convolution.size(0), -1)
            pooled_outputs.append(pooled)
        doc_embedding = torch.cat(pooled_outputs, 1)
        return doc_embedding


