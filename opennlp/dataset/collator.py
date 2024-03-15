# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-15

import torch
from opennlp.dataset import ClassificationDataset as cDataset


class ClassificationCollator(object):

    def __init__(self, config, label_size, is_multi):
        self.config = config
        self.label_size = label_size
        self.is_multi = is_multi

    def _get_multi_hot_label(self, doc_labels):
        """For multi-label encoder
        Generate multi-hot for input labels
        e.g. input: [[0,1], [2]]
             output: [[1,1,0], [0,0,1]]
        """
        batch_size = len(doc_labels)
        max_label_num = max([len(x) for x in doc_labels])
        doc_labels_extend = \
            [[doc_labels[i][0] for x in range(max_label_num)] for i in range(batch_size)]
        for i in range(0, batch_size):
            doc_labels_extend[i][0 : len(doc_labels[i])] = doc_labels[i]
        y = torch.Tensor(doc_labels_extend).long()
        y_onehot = torch.zeros(batch_size, self.label_size).scatter_(1, y, 1)
        return y_onehot

    def _append_label(self, doc_labels, sample):
        if self.is_multi:
            doc_labels.append(sample[cDataset.DOC_LABEL])
        else:
            assert len(sample[cDataset.DOC_LABEL]) == 1
            doc_labels.extend(sample[cDataset.DOC_LABEL])

    def __call__(self, batch):
        def _append_vocab(ori_vocabs, vocabs, lens, max_len=None):
            padding = [cDataset.VOCAB_PADDING] * (max_len - len(ori_vocabs))
            vocabs.append(ori_vocabs + padding)
            lens.append(len(ori_vocabs))

        doc_labels = []
        doc_tokens = []
        doc_tokens_len = []

        for _, value in enumerate(batch):
            self._append_label(doc_labels, value)
            _append_vocab(value[cDataset.DOC_TOKEN], doc_tokens,
                          doc_tokens_len, self.config.max_token_len)

        if self.is_multi:
            tensor_doc_labels = self._get_multi_hot_label(doc_labels)
            doc_label_list = doc_labels
        else:
            tensor_doc_labels = torch.tensor(doc_labels)
            doc_label_list = [[x] for x in doc_labels]

        batch_map = {
            cDataset.DOC_LABEL: tensor_doc_labels,
            cDataset.DOC_LABEL_LIST: doc_label_list,
            cDataset.DOC_TOKEN: torch.tensor(doc_tokens),
            cDataset.DOC_TOKEN_LEN: torch.tensor(doc_tokens_len, dtype=torch.float32)
        }

        return batch_map

