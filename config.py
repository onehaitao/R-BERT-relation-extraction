#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import argparse
import torch
import os
import random
import json
import numpy as np


class Config(object):
    def __init__(self):
        # get init config
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # select device
        self.device = None
        if self.cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.cuda))
        else:
            self.device = torch.device('cpu')

        # create model dir
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # PLM dir
        self.plm_dir = os.path.join(self.plm_root_dir, self.plm_name)

        # data cache dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # backup data
        self.__config_backup(args)

        # set the random seed
        self.__set_seed(self.seed)

    def __get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # several key selective parameters
        parser.add_argument('--data_dir',
                            type=str,
                            default='./data',
                            help='dir to load data')
        parser.add_argument('--output_dir',
                            type=str,
                            default='./output',
                            help='dir to save output')
        parser.add_argument('--cache_dir',
                            type=str,
                            default='./cache',
                            help='dir to save data cache')

        # train settings
        parser.add_argument('--model_name',
                            type=str,
                            default='R-BERT',
                            help='model name')
        parser.add_argument('--mode',
                            type=int,
                            default=0,
                            choices=[0, 1],
                            help='running mode: 0 for training; otherwise testing')
        parser.add_argument('--seed',
                            type=int,
                            default=1234,
                            help='random seed')
        parser.add_argument('--cuda',
                            type=int,
                            default=0,
                            help='num of gpu device, if -1, select cpu')

        # about pretrained language models (PLMs)
        parser.add_argument('--plm_root_dir',
                            type=str,
                            default='./resource',
                            help='dir to pretrained language models')
        parser.add_argument('--plm_name',
                            type=str,
                            default='bert-base-uncased',
                            help='dir to pretrained language model')

        # hyper parameters
        parser.add_argument('--epoch',
                            type=int,
                            default=5,
                            help='max epoches during training')
        parser.add_argument('--max_len',
                            type=int,
                            default=128,
                            help='max length of sentence after tokenization')
        parser.add_argument('--batch_size',
                            type=int,
                            default=16,
                            help='batch size')
        parser.add_argument('--lr',
                            type=float,
                            default=2e-5,
                            help='learning rate in PLM layers')
        parser.add_argument('--other_lr',
                            type=float,
                            default=2e-5,
                            help='learning rate in other layers except PLM')
        parser.add_argument('--weight_decay',
                            type=float,
                            default=0.0,
                            help='weight decay')
        parser.add_argument('--adam_epsilon',
                            type=float,
                            default=1e-8,
                            help='epsilon for Adam optimizer')
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=1,
                            help='number of updates steps to accumulate before \
                                performing a backward update pass')
        parser.add_argument('--warmup_proportion',
                            type=int,
                            default=0.1,
                            help='proportion of linear warmup over warmup_steps')
        parser.add_argument('--max_grad_norm',
                            type=float,
                            default=1.0,
                            help='max gradient norm')

        # other hyper parameters
        parser.add_argument('--dropout',
                            type=float,
                            default=0.1,
                            help='the possiblity of dropout')

        args = parser.parse_args()
        return args

    def __set_seed(self, seed=1234):
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # set seed for cpu
        torch.cuda.manual_seed(seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(seed)  # set seed for all gpu

    def __config_backup(self, args):
        config_backup_path = os.path.join(self.model_dir, 'user_config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False, indent=4)

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])


if __name__ == '__main__':
    config = Config()
    config.print_config()
