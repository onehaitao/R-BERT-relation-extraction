#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, CONFIG_NAME

from config import Config
from utils import RelationLoader, SemEvalDataLoader
from model import R_BERT
from evaluate import Eval


class Runner(object):
    def __init__(self, id2rel, loader, user_config):
        self.class_num = len(id2rel)
        self.id2rel = id2rel
        self.loader = loader
        self.user_config = user_config

        self.model = R_BERT(self.class_num, user_config)
        self.model = self.model.to(user_config.device)
        self.eval_tool = Eval(user_config)

    def train(self):
        train_loader, dev_loader, _ = self.loader
        num_training_steps = len(train_loader) // self.user_config.\
            gradient_accumulation_steps * self.user_config.epoch
        num_warmup_steps = int(num_training_steps *
                               self.user_config.warmup_proportion)

        bert_params = list(map(id, self.model.bert.parameters()))
        rest_params = filter(lambda p: id(
            p) not in bert_params, self.model.parameters())

        optimizer_grouped_parameters = [
            {'params': self.model.bert.parameters()},
            {'params': rest_params,  'lr': self.user_config.other_lr},
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.user_config.lr,
            eps=self.user_config.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        print('--------------------------------------')
        print('traning model parameters (except PLM layers):')
        for name, param in self.model.named_parameters():
            if id(param) in bert_params:
                continue
            if param.requires_grad:
                print('%s :  %s' % (name, str(param.data.shape)))

        print('--------------------------------------')
        print('start to train the model ...')

        max_f1 = -float('inf')
        for epoch in range(1, 1+self.user_config.epoch):
            train_loss = 0.0
            data_iterator = tqdm(train_loader, desc='Train')
            for _, (data, label) in enumerate(data_iterator):
                self.model.train()
                data = data.to(self.user_config.device)
                label = label.to(self.user_config.device)

                optimizer.zero_grad()
                loss, _ = self.model(data, label)
                train_loss += loss.item()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.user_config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()

            train_loss = train_loss / len(train_loader)

            f1, dev_loss, _ = self.eval_tool.evaluate(self.model, dev_loader)
            print('[%03d] train_loss: %.3f | dev_loss: %.3f | f1 on dev: %.4f'
                  % (epoch, train_loss, dev_loss, f1), end=' ')
            if f1 > max_f1:
                max_f1 = f1
                model_to_save = self.model.module if hasattr(
                    self.model, 'module') else self.model
                output_model_file = os.path.join(
                    self.user_config.model_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(
                    self.user_config.model_dir, CONFIG_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.bert.config.to_json_file(output_config_file)
                print('>>> save models!')
            else:
                print()

    def test(self):
        print('--------------------------------------')
        print('start load model ...')
        if not os.path.exists(self.user_config.model_dir):
            raise Exception('no pre-trained model exists!')

        state_dict = torch.load(
            os.path.join(self.user_config.model_dir, WEIGHTS_NAME),
            map_location=self.user_config.device
        )
        self.model.load_state_dict(state_dict)

        print('--------------------------------------')
        print('start test ...')
        _, _, test_loader = self.loader
        f1, test_loss, predict_label = self.eval_tool.evaluate(self.model, test_loader)
        print('test_loss: %.3f | f1 on test: %.3f' % (test_loss, f1))
        return predict_label


def print_result(predict_label, id2rel, start_idx=8001):
    des_file = './eval/predicted_result.txt'
    with open(des_file, 'w', encoding='utf-8') as fw:
        for i in range(0, predict_label.shape[0]):
            fw.write('{}\t{}\n'.format(
                start_idx+i, id2rel[int(predict_label[i])]))


if __name__ == '__main__':
    user_config = Config()
    print('--------------------------------------')
    print('some config:')
    user_config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    rel2id, id2rel, class_num = RelationLoader(user_config).get_relation()
    loader = SemEvalDataLoader(rel2id, user_config)

    train_loader, dev_loader, test_loader = None, None, None
    if user_config.mode == 0:  # train mode
        train_loader = loader.get_train()
        dev_loader = loader.get_dev()
        test_loader = loader.get_test()
    elif user_config.mode == 1:
        test_loader = loader.get_test()
    loader = [train_loader, dev_loader, test_loader]
    print('finish!')

    runner = Runner(id2rel, loader, user_config)
    if user_config.mode == 0:  # train mode
        runner.train()
        predict_label = runner.test()
    elif user_config.mode == 1:
        predict_label = runner.test()
    else:
        raise ValueError('invalid train mode!')
    print_result(predict_label, id2rel)
