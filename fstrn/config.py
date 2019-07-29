#! /usr/bin/python
# -*- coding: utf8 -*-

from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()
config.VALID = edict()
config.scale = 4
## Adam
config.TRAIN.batch_size = 21  # 512
config.TRAIN.lr = 1e-4  # start from 1e-3
config.TRAIN.beta1 = 0.9
## initialize G
config.TRAIN.lr_decrease = 1.0
config.TRAIN = 100

config.time_step = 5
config.VALID.batch_size = 1


config.TRAIN.path = '../data/train/'
config.VALID.path = '../data/test/'
