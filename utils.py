#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6


import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_bert import BertTokenizer


class RelationLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir

    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()


class Tokenizer(object):
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.plm_dir = config.plm_dir
        self.tokenizer, self.special_tokens = self.load_tokenizer()

    def load_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.plm_dir)
        # entity marker: <e1>, </e1> -> `$`, <e2>, </e2> -> `#`
        special_tokens = ['$', '#']
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer, special_tokens

    def build_vocab(self):
        vocab = set()
        filelist = ['train', 'test']
        for filename in filelist:
            src_file = os.path.join(self.data_dir, '{}.json'.format(filename))
            if not os.path.isfile(src_file):
                continue
            print('get the result of tokenization from %s' % src_file)
            with open(src_file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    sentence = json.loads(line.strip())['sentence']
                    for token in sentence:
                        if token in ['<e1>', '</e1>', '<e2>', '</e2>']:
                            continue
                        vocab.add(token)
        return vocab

    def get_vocab(self):
        vocab_set = self.build_vocab()
        vocab_dict = {}
        extra_tokens = ['[CLS]', '[SEP]', '[PAD]']
        for token in extra_tokens + self.special_tokens:
            vocab_dict[token] = [self.tokenizer.convert_tokens_to_ids(token)]

        for token in vocab_set:
            token = token.lower()
            if token in vocab_dict.keys():
                continue
            token_res = self.tokenizer.tokenize(token)
            if len(token_res) < 1:
                token_idx_list = [self.tokenizer.convert_tokens_to_ids('[UNK]')]
            else:
                token_idx_list = self.tokenizer.convert_tokens_to_ids(token_res)
            vocab_dict[token] = token_idx_list
        return vocab_dict


class SemEvalCorpus(object):
    def __init__(self, rel2id, config):
        self.rel2id = rel2id
        self.class_num = len(rel2id)
        self.max_len = config.max_len
        self.data_dir = config.data_dir
        self.cache_dir = config.cache_dir
        self.tokenizer = Tokenizer(config)
        self.vocab = None

    def __symbolize_sentence(self, sentence):
        """
            Args:
                sentence (list)
            Return:
                sent(ids): [CLS] ... $ e1 $ ... # e2 # ... [SEP] [PAD]
                mask     :   1    3  4  4 4  3  5  5 5  3    2     0
        """
        assert '<e1>' in sentence
        assert '<e2>' in sentence
        assert '</e1>' in sentence
        assert '</e2>' in sentence
        sentence_token = []
        sentence_mask = []
        # postion of e1 (p11, p12), e2 (p21, p22) after tokenization
        p11 = p12 = p21 = p22 = -1
        for token in sentence:
            token = token.lower()
            if token == '<e1>':
                p11 = len(sentence_token)
                sentence_token += self.vocab['$']
            elif token == '</e1>':
                p12 = len(sentence_token)
                sentence_token += self.vocab['$']
            elif token == '<e2>':
                p21 = len(sentence_token)
                sentence_token += self.vocab['#']
            elif token == '</e2>':
                p22 = len(sentence_token)
                sentence_token += self.vocab['#']
            else:
                bert_token = self.vocab[token]
                sentence_token += bert_token
        sentence_mask = [3] * len(sentence_token)
        sentence_mask[p11: p12+1] = [4] * (p12 - p11 + 1)
        sentence_mask[p21: p22+1] = [5] * (p22 - p21 + 1)

        if len(sentence_token) > self.max_len-2:
            sentence_token = sentence_token[:self.max_len-2]
            sentence_mask = sentence_mask[:self.max_len-2]

        pad_length = self.max_len - 2 - len(sentence_token)
        mask = [1] + sentence_mask + [2] + [0] * pad_length
        input_ids = self.vocab['[CLS]'] + sentence_token + self.vocab['[SEP]']
        input_ids += self.vocab['[PAD]'] * pad_length

        assert len(mask) == self.max_len
        assert len(input_ids) == self.max_len

        unit = np.asarray([input_ids, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 2, self.max_len))
        return unit

    def __load_data(self, filetype):
        data_cache = os.path.join(self.cache_dir, '{}.pkl'.format(filetype))
        if os.path.exists(data_cache):
            data, labels = torch.load(data_cache)
        else:
            if self.vocab is None:
                self.vocab = self.tokenizer.get_vocab()
            src_file = os.path.join(self.data_dir, '{}.json'.format(filetype))
            data = []
            labels = []
            with open(src_file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    line = json.loads(line.strip())
                    label = line['relation']
                    sentence = line['sentence']
                    label_idx = self.rel2id[label]
                    one_sentence = self.__symbolize_sentence(sentence)
                    data.append(one_sentence)
                    labels.append(label_idx)
            data_labels = [data, labels]
            torch.save(data_labels, data_cache)
        return data, labels

    def load_corpus(self, filetype):
        """
        filetype:
            train: load training data
            test : load testing data
            dev  : load development data
        """
        if filetype in ['train', 'dev', 'test']:
            return self.__load_data(filetype)
        else:
            raise ValueError('mode error!')


class SemEvalDateset(Dataset):
    def __init__(self, data, labels):
        self.dataset = data
        self.label = labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


class SemEvalDataLoader(object):
    def __init__(self, rel2id, config):
        self.rel2id = rel2id
        self.config = config
        self.corpus = SemEvalCorpus(rel2id, config)

    def __collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data, label

    def __get_data(self, filetype, shuffle=False):
        data, labels = self.corpus.load_corpus(filetype)
        dataset = SemEvalDateset(data, labels)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=self.__collate_fn
        )
        return loader

    def get_train(self):
        ret = self.__get_data(filetype='train', shuffle=True)
        print('finish loading train!')
        return ret

    def get_dev(self):
        ret = self.__get_data(filetype='test', shuffle=False)
        print('finish loading dev!')
        return ret

    def get_test(self):
        ret = self.__get_data(filetype='test', shuffle=False)
        print('finish loading test!')
        return ret


if __name__ == '__main__':
    from config import Config
    config = Config()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    print('class_num', class_num)

    loader = SemEvalDataLoader(rel2id, config)
    test_loader = loader.get_test()

    for step, (data, label) in enumerate(test_loader):
        print(type(data), data.shape)
        print(type(label), label.shape)
        import pdb
        pdb.set_trace()
        break

    train_loader = loader.get_train()
    dev_loader = loader.get_dev()
