# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-22

from torch import nn
from opennlp.embedding.embedding_layer import EmbeddingLayer
from opennlp.encoder.encode_layer import EncodeLayer
from opennlp.decoder.decode_layer import DecodeLayer


class GenerateModel(nn.Module):
    def __init__(self, dataset, config, logger):
        super(GenerateModel, self).__init__()
        self.config = config
        self.padding_idx = dataset.VOCAB_PADDING
        self.embedding = EmbeddingLayer(dataset, config, logger)
        self.encoder = EncodeLayer(config)
        label_size = len(dataset.label_map)
        hidden_size = self.encoder.get_hidden_size()
        self.decoder = DecodeLayer(config, hidden_size, label_size)

    def get_optimize_parameters(self):
        params = self.embedding.get_optimize_parameters()
        params.extend(self.encoder.get_optimize_parameters())
        params.extend(self.decoder.get_optimize_parameters())
        return params

    def forward(self, vocab_ids, **kwargs):
        embedding = self.embedding(vocab_ids, padding_idx=self.padding_idx)
        doc_embedding = self.encoder(embedding, **kwargs)
        output = self.decoder(doc_embedding, **kwargs)
        return output


