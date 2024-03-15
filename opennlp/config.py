# -*- coding:utf-8 -*-
import os
import json
import yaml


class dict2obj(dict):
    def __init__(self, *args, **kwargs):
        super(dict2obj, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = dict2obj(value)
        return value


class ConfigEngine(object):
    def __init__(self, logger, config_file=None, **kwargs):
        self.logger = logger
        self.status = True
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), "config.yml")

        with open(config_file, 'r', encoding='utf-8') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)

        self.parameters = config.get("parameters")
        if kwargs:
            self.override_parameters(**kwargs)
        self.parse_parameters()

    def __repr__(self):
        args = json.dumps(self.parameters, indent=4, ensure_ascii=False)
        self.logger.info(args)

    def validate_parameter(self, key, parameter):
        required = parameter.get("required")
        type_ = parameter.get("type")
        value = parameter.get("value")
        range_ = parameter.get("range")
        if required:
            if value is None:
                self.logger.error(f"<{key}> value <{parameter['value']}> missing.")
                return False

            if type_ == 'created':
                if os.path.exists(value):
                    self.logger.error(f"<{key}> directory <{value}> is exist.")
                    return False
            elif type_ == 'file':
                if not os.path.exists(value):
                    self.logger.error(f"<{key}> directory <{value}> does not exist.")
                    return False
            elif type_ == 'int_list':
                assert type(value) == list and type(value[0]) == int
            elif type_ == 'list':
                if type(value) != type(range_[0]):
                    self.logger.error(f"<{key}> value {type(value)} and type {type(range_[0])} do not match.")
                    return False
                if value not in range_:
                    self.logger.error(f"<{key}> value <{value}> range <{range_}> is out of bounds.")
                    return False
            else:
                if type_ == 'int':
                    if type(value) != int:
                        self.logger.error(f"<{key}> value <{type(value)}> and type <{type_}> do not match.")
                        return False
                    if value < range_[0] or value > range_[1]:
                        self.logger.error(f"<{key}> value <{value}> range <{range_}> is out of bounds.")
                        return False
                elif type_ == 'float':
                    if type(value) != float:
                        self.logger.error(f"<{key}> value <{type(value)}> and type <{type_}> do not match.")
                        return False
                    if value < range_[0] or value > range_[1]:
                        self.logger.error(f"<{key}> value <{value}> range <{range_}> is out of bounds.")
                        return False
                elif type_ == 'str':
                    if type(value) != str:
                        self.logger.error(f"<{key}> value <{type(value)}> and type <{type_}> do not match.")
                        return False
                else:
                    self.logger.error(f"<{key}> type <{type_}> does not exist.")
                    return False

        return True

    def override_parameters(self, **kwargs):
        for category, args in kwargs.items():
            if self.parameters.get(category) is None:
                self.logger.error(f"Not parameter category: {category}")
                continue
            for key, val in args.items():
                if self.parameters[category].get(key) is None:
                    self.logger.error(f"Not parameter: {key}")
                    continue
                self.parameters[category][key]['value'] = val
                self.logger.info(f"update parameter: {category}.{key}={val}")

    def parse_parameters(self):
        self.dict = dict()
        for c, args in self.parameters.items():
            if self.dict.get(c) is None:
                self.dict[c] = dict()
            for k, v in args.items():
                # if not self.validate_parameter(k, v):
                #     self.status = False
                self.dict[c][k] = v.get('value')

    def get_parameters(self):
        return dict2obj(self.dict)

    def get_status(self):
        return self.status




if __name__ == "__main__":
    from opennlp.logger import Logger
    logger = Logger(log_file='../utils/logger.txt')
    kwargs = {"data":
                  {"ngram": 3,
                   "trainset": "xxxx"},
              "feature": {}
              }
    config = ConfigEngine(logger,
                          "../data/kconfig/config.yml",
                          **kwargs)
    #config.__repr__()
    args = config.get_parameters()
    print(args.data.dict_dir)
