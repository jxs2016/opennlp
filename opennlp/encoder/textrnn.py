# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-20

import torch
from torch import nn
from opennlp.functional.rnn import RNN


class DocEmbeddingType:
    """Standard names for doc embed type.
    """
    AVG = 'AVG'
    ATTENTION = 'Attention'
    LAST_HIDDEN = 'LastHidden'

    @classmethod
    def str(cls):
        return ",".join(
            [cls.AVG, cls.ATTENTION, cls.LAST_HIDDEN])


class SumAttention(torch.nn.Module):
    """
        Reference: Hierarchical Attention Networks for Document Classification
    """

    def __init__(self, input_dimension, attention_dimension):
        super(SumAttention, self).__init__()
        self.attention_matrix = nn.Linear(input_dimension, attention_dimension)
        self.attention_vector = nn.Linear(attention_dimension, 1, bias=False)
        #init_tensor(self.attention_matrix.weight, args.initializer)
        #init_tensor(self.attention_vector.weight, args.initializer)
        # self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, inputs):
        if inputs.size(1) == 1:
            return inputs.squeeze()
        u = torch.tanh(self.attention_matrix(inputs))
        v = self.attention_vector(u)
        #alpha = torch.nn.functional.softmax(v, 1).squeeze().unsqueeze(1) #TODO 当最后一个batch_size=1时，报错
        alpha = torch.softmax(v, 1).squeeze(2).unsqueeze(1)
        score = torch.matmul(alpha, inputs).squeeze()
        if score.dim() == 1:
            score = score.unsqueeze(0)
        return score


class TextRNN(nn.Module):
    """
    Implement TextRNN, contains LSTM，BiLSTM，GRU，BiGRU
    Reference: "Effective LSTMs for Target-Dependent Sentiment Classification"
               "Bidirectional LSTM-CRF Models for Sequence Tagging"
               "Generative and discriminative text encoder
                with recurrent neural networks"
    """

    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.config = config
        self.doc_embedding_type = config.TextRNN.context_type
        self.rnn = RNN(config.embedding.dimension,
                       config.TextRNN.hidden_dimension,
                       num_layers=config.TextRNN.num_layers,
                       bidirectional=config.TextRNN.bidirectional,
                       rnn_type=config.TextRNN.rnn_type,
                       batch_first=True)
        self.hidden_size = config.TextRNN.hidden_dimension
        if config.TextRNN.bidirectional:
            self.hidden_size *= 2
        self.sum_attention = SumAttention(self.hidden_size,
                                          config.TextRNN.attention_dimension)

    def get_optimize_parameters(self):
        params = list()
        params.append({'params': self.rnn.parameters()})
        if self.doc_embedding_type == DocEmbeddingType.ATTENTION:
            params.append({'params': self.sum_attention.parameters()})
        return params

    def forward(self, embedding, **kwargs):
        input_lengths = kwargs.get("input_lengths")
        assert input_lengths is not None, "input_lengths should not be None."
        output, last_hidden = self.rnn(embedding, input_lengths)
        if self.doc_embedding_type == DocEmbeddingType.AVG:
            doc_embedding = torch.sum(output, 1) / input_lengths.unsqueeze(1)
        elif self.doc_embedding_type == DocEmbeddingType.ATTENTION:
            doc_embedding = self.sum_attention(output)
        elif self.doc_embedding_type == DocEmbeddingType.LAST_HIDDEN:
            doc_embedding = last_hidden
        else:
            raise TypeError(
                "Unsupported rnn init type: %s. Supported rnn type is: %s" % (
                    self.doc_embedding_type, DocEmbeddingType.str()))

        return doc_embedding




