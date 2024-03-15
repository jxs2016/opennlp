# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-15

import os
import json
from torch.utils.data import dataset
from opennlp.util import ModeType


class ClassificationDataset(dataset.Dataset):
    CHARSET = "utf-8"
    VOCAB_PADDING = 0
    VOCAB_UNKNOWN = 1
    MAX_LABEL_SIZE = 100000

    DOC_LABEL = "doc_label"
    DOC_TOKEN = "doc_token"
    DOC_TOKEN_LEN = "doc_token_length"
    DOC_LABEL_LIST = "doc_label_list"
    CLASSIFICATION_LABEL_SEPARATOR = "--"

    def __init__(self, config, json_files, logger, mode, generate_dict=False):
        self.config = config
        self.logger = logger
        self.model_mode = mode
        self._init_dict()
        self.hierarchy_classes = []
        self.sample_index = []
        self.sample_size = 0
        self.files = json_files
        if self.files:
            self.get_sample(self.files)
        if generate_dict:
            self.generate_feature_dict()
        self._load_dict()

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        if idx >= self.sample_size:
            raise IndexError
        index = self.sample_index[idx]
        with open(self.files, "r", encoding=self.CHARSET) as fin:
            fin.seek(index)
            json_str = fin.readline()
            json_str = json.loads(json_str)
        return self._get_vocab_id_list(json_str)

    def _init_dict(self):
        self.dict_names = [self.DOC_LABEL, self.DOC_TOKEN]
        # By default keep all labels
        self.min_count = [0, self.config.min_token_count]
        self.max_dict_size = [self.MAX_LABEL_SIZE, self.config.max_token_dict_size]
        self.max_sequence_length = [self.config.max_token_len]
        # When generating dict, the following map store vocab count.
        # Then clear dict and load vocab of word index
        self.label_map = dict()
        self.token_map = dict()
        self.dicts = [self.label_map, self.token_map]

        # Save sorted dict according to the count
        self.label_count_list = []
        self.token_count_list = []
        self.count_list = [self.label_count_list, self.token_count_list]

        self.id_to_label_map = dict()
        self.id_to_token_map = dict()
        self.id_to_vocab_dict_list = [self.id_to_label_map, self.id_to_token_map]

        self.dict_files = []
        for dict_name in self.dict_names:
            dict_path = os.path.join(self.config.dict_dir, dict_name + ".dict")
            self.dict_files.append(dict_path)
        self.label_dict_file = self.dict_files[0]

        if self.config.pretrained_embedding_filepath and \
                os.path.exists(self.config.pretrained_embedding_filepath):
            self.pretrained_dict_names = [self.DOC_TOKEN]
            self.pretrained_dict_files = [self.config.pretrained_embedding_filepath]
            self.pretrained_min_count = [self.config.min_token_count]

    def _insert_vocab(self, files):
        for _, file in enumerate(files):
            with open(file, "r", encoding=self.CHARSET) as _fin:
                for _json_str in _fin:
                    try:
                        _json_str = json.loads(_json_str)
                        _json_str = self._preprocess_obj(_json_str, ngram=self.config.ngram)
                        self._insert_sequence_vocab(_json_str[self.DOC_LABEL], self.label_map)
                        self._insert_sequence_vocab(_json_str[self.DOC_TOKEN], self.token_map)
                    except:
                        self.logger.warn(_json_str)

    def get_sample(self, file):
        with open(file, "r", encoding=self.CHARSET) as fin:
            self.sample_index.append(0)
            while True:
                try:
                    json_str = fin.readline()
                    if not json_str:
                        self.sample_index.pop()
                        break
                    self.sample_size += 1
                    self.sample_index.append(fin.tell())
                except Exception as e:
                    self.logger.error(e)

    def generate_feature_dict(self):
        vocab_json_files = [self.config.trainset_dir, self.config.validateset_dir]
        if self.config.testset_dir and os.path.exists(self.config.testset_dir):
            vocab_json_files.append(self.config.testset_dir)
        self.logger.info(f"Use dataset to generate dict: {vocab_json_files}")
        self._insert_vocab(vocab_json_files)

        if self.config.pretrained_embedding_filepath and \
                os.path.exists(self.config.pretrained_embedding_filepath):
            self.logger.info(
                f"Use pretrained embed to generate dict: {self.config.pretrained_embedding_filepath}")
            self._load_pretrained_dict()

        self._print_dict_info()
        self._shrink_dict()
        self.logger.info("Shrink dict over.")
        self._print_dict_info(True)
        self._save_dict()
        self._clear_dict()

    def _save_dict(self, dict_name=None):
        """Save vocab to file and generate id_to_vocab_dict_map
        Args:
            dict_name: Dict name, if None save all dict. Default None.
        """
        if dict_name is None:
            if not os.path.exists(self.config.dict_dir):
                os.makedirs(self.config.dict_dir)
            for name in self.dict_names:
                self._save_dict(name)
        else:
            dict_idx = self.dict_names.index(dict_name)
            dict_file = open(self.dict_files[dict_idx], "w", encoding=self.CHARSET)
            id_to_vocab_dict_map = self.id_to_vocab_dict_list[dict_idx]
            index = 0
            for vocab, count in self.count_list[dict_idx]:
                id_to_vocab_dict_map[index] = vocab
                index += 1
                dict_file.write("%s\t%d\n" % (vocab, count))
            dict_file.close()

    def _load_dict(self, dict_name=None):
        """Load dict from file.
        Args:
            dict_name: Dict name, if None load all dict. Default None.
        Returns:
            dict.
        """
        if dict_name is None:
            for name in self.dict_names:
                self._load_dict(name)
        else:
            dict_idx = self.dict_names.index(dict_name)
            if not os.path.exists(self.dict_files[dict_idx]):
                self.logger.warn("Not exists %s for %s" % (
                    self.dict_files[dict_idx], dict_name))
            else:
                dict_map = self.dicts[dict_idx]
                id_to_vocab_dict_map = self.id_to_vocab_dict_list[dict_idx]
                if dict_name != self.DOC_LABEL:
                    dict_map[self.VOCAB_PADDING] = 0
                    dict_map[self.VOCAB_UNKNOWN] = 1
                    id_to_vocab_dict_map[0] = self.VOCAB_PADDING
                    id_to_vocab_dict_map[1] = self.VOCAB_UNKNOWN

                    for line in open(self.dict_files[dict_idx], "r", encoding=self.CHARSET):
                        vocab = line.strip("\n").split("\t")
                        dict_idx = len(dict_map)
                        dict_map[vocab[0]] = dict_idx
                        id_to_vocab_dict_map[dict_idx] = vocab[0]
                else:
                    hierarchy_dict = dict()
                    for line in open(self.dict_files[dict_idx], "r", encoding=self.CHARSET):
                        vocab = line.strip("\n").split("\t")
                        dict_idx = len(dict_map)
                        dict_map[vocab[0]] = dict_idx
                        id_to_vocab_dict_map[dict_idx] = vocab[0]

                        k_level = len(vocab[0].split(self.CLASSIFICATION_LABEL_SEPARATOR))
                        if k_level not in hierarchy_dict:
                            hierarchy_dict[k_level] = [vocab[0]]
                        else:
                            hierarchy_dict[k_level].append(vocab[0])
                    sorted_hierarchy_dict = sorted(hierarchy_dict.items(), key=lambda r: r[0])
                    for _, level_dict in sorted_hierarchy_dict:
                        self.hierarchy_classes.append(len(level_dict))

    def _load_pretrained_dict(self, dict_name=None, pretrained_file=None, min_count=0):
        """Use pretrained embed to generate dict
        """
        if dict_name is None:
            for i, _ in enumerate(self.pretrained_dict_names):
                self._load_pretrained_dict(
                    self.pretrained_dict_names[i],
                    self.pretrained_dict_files[i],
                    self.pretrained_min_count[i])

        else:
            index = self.dict_names.index(dict_name)
            dict_map = self.dicts[index]
            with open(pretrained_file, "r", encoding=self.CHARSET) as fin:
                for line in fin:
                    data = line.strip().split(' ')
                    if len(data) == 2:
                        continue
                    if data[0] not in dict_map:
                        dict_map[data[0]] = 0
                    dict_map[data[0]] += min_count + 1

    def _shrink_dict(self, dict_name=None):
        if dict_name is None:
            for name in self.dict_names:
                self._shrink_dict(name)
        else:
            dict_idx = self.dict_names.index(dict_name)
            self.count_list[dict_idx] = sorted(self.dicts[dict_idx].items(),
                                               key=lambda x: (x[1], x[0]),
                                               reverse=True)
            self.count_list[dict_idx] = \
                [(k, v) for k, v in self.count_list[dict_idx] if
                 v >= self.min_count[dict_idx]][0:self.max_dict_size[dict_idx]]

    def _clear_dict(self):
        """Clear all dict
        """
        for dict_map in self.dicts:
            dict_map.clear()
        for id_to_vocab_dict in self.id_to_vocab_dict_list:
            id_to_vocab_dict.clear()

    def _print_dict_info(self, count_list=False):
        """Print dict info
        """
        for i, dict_name in enumerate(self.dict_names):
            if count_list:
                self.logger.info(
                    "Size of %s dict is %d" % (
                        dict_name, len(self.count_list[i])))
            else:
                self.logger.info(
                    "Size of %s dict is %d" % (dict_name, len(self.dicts[i])))

    def _insert_sequence_vocab(self, sequence_vocabs, dict_map):
        for vocab in sequence_vocabs:
            self._add_vocab_to_dict(dict_map, vocab)

    @staticmethod
    def _add_vocab_to_dict(dict_map, vocab):
        if vocab not in dict_map:
            dict_map[vocab] = 0
        dict_map[vocab] += 1

    def _label_to_id(self, sequence_labels, dict_map):
        """Convert label to id. The reason that label is not in label map may be
        label is filtered or label in validate/test does not occur in train set
        """
        label_id_list = []
        for label in sequence_labels:
            if label not in dict_map:
                self.logger.warn("Label not in label map: %s" % label)
            else:
                label_id_list.append(self.label_map[label])
        assert label_id_list, "Label is empty: %s" % " ".join(sequence_labels)

        return label_id_list

    def _vocab_to_id(self, sequence_vocabs, dict_map, skip_unk=False):
        """
            Convert vocab to id. Vocab not in dict map will be map to _UNK
        """
        vocab_id_list = []
        for x in sequence_vocabs:
            idx = dict_map.get(x, self.VOCAB_UNKNOWN)
            if skip_unk and idx == self.VOCAB_UNKNOWN:
                continue
            vocab_id_list.append(idx)
        if not vocab_id_list:
            vocab_id_list.append(self.VOCAB_PADDING)
        return vocab_id_list

    def _get_vocab_id_list(self, json_obj):
        """
          Use dict to convert all vocabs to ids
        """
        json_obj = self._preprocess_obj(json_obj, ngram=self.config.ngram)
        doc_labels = json_obj[self.DOC_LABEL]
        doc_tokens = json_obj[self.DOC_TOKEN]
        token_ids = self._vocab_to_id(doc_tokens, self.token_map, self.config.skip_unk)

        dataset = {self.DOC_LABEL: self._label_to_id(doc_labels, self.label_map)
                                   if self.model_mode != ModeType.PREDICT else [0],
                   self.DOC_TOKEN: token_ids}
        return dataset

    def _preprocess_obj(self, json_obj, **kwargs):
        ngram = kwargs.get('ngram', 0)
        if ngram > 1:
            tokens = json_obj[self.DOC_TOKEN]
            ngram_tokens = []
            for j in range(2, ngram + 1):
                ngrams = [" ".join(tokens[k:k + j]) for k in range(len(tokens) - j + 1)]
                ngram_tokens.extend(ngrams)
            json_obj[self.DOC_TOKEN] = ngram_tokens

        json_obj[self.DOC_TOKEN] = json_obj[self.DOC_TOKEN][0:self.config.max_token_len]
        return json_obj
