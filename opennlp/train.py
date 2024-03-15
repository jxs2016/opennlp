# -*- coding:utf-8 -*-
# -*- coding:utf-8 -*-
# author: amos.ji
# created: 2021-10-22

import os
import shutil
import time
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from opennlp.dataset import ClassificationDataset
from opennlp.dataset import ClassificationCollator
from opennlp.evaluate import ClassificationEvaluator as cEvaluator
from opennlp.functional.loss import ClassificationLoss
from opennlp.optimizer.optimizer import Optimizer
from opennlp.util import ModeType, TaskType
from opennlp.generate_model import GenerateModel


def get_data_loader(config, logger, is_multi, use_test=False):
    """
        Get data loader: Train, Validate, Test
    """
    train_dataset = ClassificationDataset(config.data,
                                          config.data.trainset_dir,
                                          logger,
                                          ModeType.TRAIN,
                                          generate_dict=True)
    collate_fn = ClassificationCollator(config.data, len(train_dataset.label_map), is_multi=is_multi)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=config.train.batch_size,
                                   collate_fn=collate_fn,
                                   shuffle=True, pin_memory=True, drop_last=True)

    validate_dataset = ClassificationDataset(config.data,
                                            config.data.validateset_dir,
                                            logger,
                                            ModeType.TRAIN,
                                            generate_dict=False)
    validate_data_loader = DataLoader(validate_dataset,
                                      batch_size=config.train.batch_size,
                                      collate_fn=collate_fn,
                                      shuffle=False, pin_memory=True, drop_last=True)
    if use_test:
        test_dataset = ClassificationDataset(config.data,
                                             config.data.testset_dir,
                                             logger,
                                             ModeType.TRAIN,
                                             generate_dict=False)
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=config.train.batch_size,
                                      collate_fn=collate_fn,
                                      shuffle=False, pin_memory=True)
        return train_data_loader, validate_data_loader, test_data_loader
    return train_data_loader, validate_data_loader, None


def get_model(dataset, config, logger):
    """
        Get encoder util from configuration
    """
    model = GenerateModel(dataset, config, logger)
    model = model.to(config.task.device)
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
    return model


class ClassificationTrainer(object):
    def __init__(self, label_map, logger, evaluator, conf, loss_fn):
        self.label_map = label_map
        self.logger = logger
        self.evaluator = evaluator
        self.conf = conf
        self.loss_fn = loss_fn

    def train(self, data_loader, model, optimizer, stage, epoch):
        model.train()
        return self.run(data_loader, model, optimizer, stage, epoch, ModeType.TRAIN)

    def eval(self, data_loader, model, optimizer, stage, epoch):
        model.eval()
        return self.run(data_loader, model, optimizer, stage, epoch)

    def run(self, data_loader, model, optimizer, stage, epoch, mode=ModeType.EVAL):
        is_multi = False
        # multi-label classifcation
        if self.conf.task.type == TaskType.MultilabelClassification:
            is_multi = True
        predict_probs = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        for batch in data_loader:
            label_ids = batch[ClassificationDataset.DOC_LABEL].to(self.conf.task.device)
            label_list = batch[ClassificationDataset.DOC_LABEL_LIST]
            input_ids = batch[ClassificationDataset.DOC_TOKEN].to(self.conf.task.device)
            input_lengths = batch[ClassificationDataset.DOC_TOKEN_LEN].to(self.conf.task.device)
            kwargs = {"input_lengths": input_lengths}
            logits = model(input_ids, **kwargs)
            loss = self.loss_fn(logits, label_ids, is_multi)
            if mode == ModeType.TRAIN:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                continue
            total_loss += loss.item()
            if not is_multi:
                result = torch.nn.functional.softmax(logits, dim=1).cpu().tolist()
            else:
                result = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(result)
            standard_labels.extend(label_list)

        if mode == ModeType.EVAL:
            total_loss = total_loss / num_batch
            (_, accuracy_list, precision_list, recall_list, fscore_list, right_list,
             predict_list, standard_list) = \
                self.evaluator.evaluate(
                    predict_probs, standard_label_ids=standard_labels, label_map=self.label_map,
                    threshold=self.conf.eval.threshold, top_k=self.conf.eval.top_k, is_multi=is_multi)
            # precision_list[0] save metrics of flat encoder
            # precision_list[1:] save metrices of hierarchical encoder
            accuracy = accuracy_list[0][cEvaluator.MICRO_AVERAGE]
            precision = precision_list[0][cEvaluator.MICRO_AVERAGE]
            recall = recall_list[0][cEvaluator.MICRO_AVERAGE]
            fscore = fscore_list[0][cEvaluator.MICRO_AVERAGE]
            macro_fscore = fscore_list[0][cEvaluator.MACRO_AVERAGE]
            right = right_list[0][cEvaluator.MICRO_AVERAGE]
            predict = predict_list[0][cEvaluator.MICRO_AVERAGE]
            standard = standard_list[0][cEvaluator.MICRO_AVERAGE]
            self.logger.warn(
                "%s performance at epoch %d is  accuracy: %f, precision: %f, "
                "recall: %f, fscore: %f, macro-fscore: %f, right: %d, predict: %d, standard: %d.\n"
                "Loss is: %f." % (stage, epoch,
                                  accuracy, precision, recall, fscore, macro_fscore,
                                  right, predict, standard,
                                  total_loss))

            if stage.startswith("BEST"):
                self.evaluator.save(eval_name=stage.replace(" ", "_"))
            return accuracy, precision, recall, fscore, macro_fscore, right, predict, standard, total_loss


def load_checkpoint(file_name, conf, model, optimizer):
    checkpoint = torch.load(file_name)
    conf.train.start_epoch = checkpoint["epoch"]
    best_performance = checkpoint["best_performance"]
    if hasattr(model, "module"):
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return best_performance


def save_checkpoint(state, file_prefix):
    file_name = file_prefix + "_" + str(state["epoch"])
    torch.save(state, file_name)


def set_seed(seed=12345):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train(conf, logger, average="macro"):
    if not os.path.exists(conf.task.checkpoint_dir):
        os.makedirs(conf.task.checkpoint_dir)
    logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
    model_name = conf.task.model_name

    use_test = False
    if conf.data.testset_dir and os.path.exists(conf.data.testset_dir):
        use_test = True
    is_multi = False
    if conf.task.type == TaskType.MultilabelClassification:
        is_multi = True

    train_data_loader, validate_data_loader, test_data_loader = get_data_loader(conf, logger, is_multi, use_test=use_test)
    empty_dataset = ClassificationDataset(conf.data, None, logger, ModeType.TRAIN)
    model = get_model(empty_dataset, conf, logger)
    loss_fn = ClassificationLoss(label_size=len(empty_dataset.label_map), loss_type=conf.train.loss_type)
    optimizer = Optimizer(conf.optimizer, model).get_optimizer()
    evaluator = cEvaluator(conf.eval.metric_dir, greedy=conf.eval.greedy)
    trainer = ClassificationTrainer(empty_dataset.label_map, logger, evaluator, conf, loss_fn)

    best_epoch = -1
    best_performance = 0
    if isinstance(model_name, list):
        model_name = "_".join(model_name)
    model_file_prefix = conf.task.checkpoint_dir + "/" + model_name
    for epoch in range(conf.train.epochs):
        start_time = time.time()
        trainer.train(train_data_loader, model, optimizer, "TRAIN", epoch)
        trainer.eval(train_data_loader, model, optimizer, "TRAIN", epoch)

        metric = trainer.eval(validate_data_loader, model, optimizer, "VALIDATE", epoch)
        if use_test:
            trainer.eval(test_data_loader, model, optimizer, "TEST", epoch)
        validate_accuracy, validate_precision, validate_recall, \
        validate_fscore, validate_macro_fscore, validate_right, validate_predict, \
        validate_standard, validate_loss = metric
        performance = validate_macro_fscore if average == "macro" else validate_fscore
        if performance > best_performance:  # record the best util
            best_epoch = epoch
            best_performance = performance
            save_checkpoint({
                        'epoch': epoch,
                        'model_name': model_name,
                        'state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                        'best_performance': best_performance,
                        'optimizer': optimizer.state_dict(),
                    }, model_file_prefix)
        time_used = time.time() - start_time
        logger.info("Epoch %d cost time: %d second" % (epoch, time_used))
    if best_epoch == -1:
        return model_name, "best epoch: -1"
    # best util on validateion set
    best_epoch_file_name = model_file_prefix + "_" + str(best_epoch)
    best_file_name = model_file_prefix + "_best"
    shutil.copyfile(best_epoch_file_name, best_file_name)
    load_checkpoint(model_file_prefix + "_" + str(best_epoch), conf, model, optimizer)

    trainer.eval(train_data_loader, model, optimizer, "BEST TRAIN", best_epoch)
    if use_test:
        metric = trainer.eval(test_data_loader, model, optimizer, "BEST TEST", best_epoch)
    else:
        metric = trainer.eval(validate_data_loader, model, optimizer, "BEST VALIDATE", best_epoch)

    accuracy, precision, recall, fscore, macro_fscore, right, predict, standard, loss = metric
    metric = {"accuracy": accuracy, "precision": precision, "recall": recall,
              "micro_fscore": fscore, "macro_fscore": macro_fscore}
    return metric
