# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-16

import numpy as np
import torch
import torch.nn as nn


class EmbeddingType:
    EMBEDDING = 'embedding'
    
    @classmethod
    def str(cls):
        return ",".join([cls.EMBEDDING])


class EmbeddingProcessType:
    FLAT = 'flat'
    MEAN = 'mean'
    SUM = 'sum'
    
    @classmethod
    def str(cls):
        return ",".join([cls.FLAT, cls.MEAN, cls.SUM])


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 args, logger, padding_idx=None):
        super(Embedding, self).__init__()
        self.logger = logger
        self.mode = args.process_type
        self.dropout = nn.Dropout(p=args.dropout)
        if self.mode == EmbeddingProcessType.FLAT:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        else:
            self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode=self.mode)

    def forward(self, vocab_ids, offset=None):
        if self.mode == EmbeddingProcessType.FLAT:
            embedding = self.embedding(vocab_ids)
        else:
            embedding = self.embedding(vocab_ids, offset)
        return self.dropout(embedding)
