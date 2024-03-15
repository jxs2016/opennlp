# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-16

import os
import torch
from torch import nn
from opennlp.embedding import Embedding
from opennlp.embedding import EmbeddingType, EmbeddingProcessType
from opennlp.functional.model_util import init_tensor


class ModeType:
    """Standard names for util modes.
    The following standard keys are defined:
    * `TRAIN`: training mode.
    * `EVAL`: evaluation mode.
    * `PREDICT`: inference mode.
    """
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'infer'

    @classmethod
    def str(cls):
        return ",".join([cls.TRAIN, cls.EVAL, cls.PREDICT])

    @classmethod
    def lists(cls):
        return [cls.TRAIN, cls.EVAL, cls.PREDICT]


class EmbeddingLayer(nn.Module):
    def __init__(self, dataset, config, logger):
        super(EmbeddingLayer, self).__init__()
        self.logger = logger
        self.padding_idx = dataset.VOCAB_PADDING
        self.embedding_dim = config.embedding.dimension
        self.embedding_type = config.embedding.embedding_type
        self.process_type = config.embedding.process_type
        vocab_map = dataset.token_map
        self.vocab_size = len(vocab_map)
        self.embedding = Embedding(self.vocab_size, self.embedding_dim,
                                    config.embedding, logger,
                                    padding_idx=self.padding_idx)
        lookup_table = torch.empty(self.vocab_size, self.embedding_dim)
        embedding_lookup_table = init_tensor(lookup_table, args=config.initializer)
        if dataset.model_mode == ModeType.TRAIN:
            if config.data.pretrained_embedding_filepath and \
                        os.path.exists(config.data.pretrained_embedding_filepath):
                self.load_pretrained_embedding(embedding_lookup_table, vocab_map,
                                               config.data.pretrained_embedding_filepath)
        if self.padding_idx is not None:
            embedding_lookup_table[self.padding_idx] = 0.0

        self.embedding.embedding.weight.data.copy_(embedding_lookup_table)

    def load_pretrained_embedding(self, embedding_lookup_table, dict_map, pretrained_embedding_file):
        self.logger.warn(f"Load embed from {pretrained_embedding_file}")
        with open(pretrained_embedding_file, encoding="utf-8") as fin:
            num_pretrained = 0
            for line in fin:
                data = line.strip().split(' ')
                # Check embed info
                if len(data) == 2:
                    assert int(data[1]) == self.embedding_dim, \
                        "Pretrained embed dim not matching: %s, %d" % (
                            data[1], self.embedding_dim)
                    continue
                if data[0] not in dict_map:
                    continue
                embedding = torch.FloatTensor([float(i) for i in data[1:]])
                embedding_lookup_table[dict_map[data[0]]] = embedding
                num_pretrained += 1
        self.logger.warn(f"Total dict size of %s is {self.vocab_size}")
        self.logger.warn(f"Size of pretrained embed is {num_pretrained}")
        self.logger.warn(f"Size of randomly initialize embed is {self.vocab_size - num_pretrained}")

    def get_optimize_parameters(self):
        params = [
            {
                'params': self.embedding.parameters(),
                'is_embedding': True
            }
        ]
        return params

    def forward(self, vocab_ids, **kwargs):
        if self.process_type == EmbeddingProcessType.FLAT:
            embedding = self.embedding(vocab_ids)
        else:
            offset = kwargs.get("offset", None)
            embedding = self.embedding(vocab_ids, offset)

        return embedding






