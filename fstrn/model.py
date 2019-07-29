#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np


def fstrn(inputs, is_train=False, reuse=False):

    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)

        I_LR = InputLayer(inputs, name='input_level')

        F_L0 = Conv3dLayer(I_LR, act=tf.nn.leaky_relu, shape=(3, 3, 3, 1, 64), strides=(1, 1, 1, 1, 1), padding='SAME',
                           W_init=tf.contrib.layers.xavier_initializer(), name='LFENet')

        FRB_Ld = F_L0
        # residual blocks
        for i in range(5):
            frb = PReluLayer(FRB_Ld, name='prelu_D%s' % (i))
            frb = Conv3dLayer(frb, shape=(1, 3, 3, 64, 64), strides=(1, 1, 1, 1, 1), padding='SAME',
                              W_init=tf.contrib.layers.xavier_initializer(), name='frb_k133%s' % (i))
            frb = Conv3dLayer(frb, shape=(3, 1, 1, 64, 64), strides=(1, 1, 1, 1, 1), padding='SAME',
                              W_init=tf.contrib.layers.xavier_initializer(), name='frb_k311%s' % (i))
            frb = ElementwiseLayer([FRB_Ld, frb], tf.add, name='frb_residual_add/%s' % i)
            FRB_Ld = frb

        LRL = ElementwiseLayer(layers=[F_L0, FRB_Ld], combine_fn=tf.add, name='LRL')
        LRL = PReluLayer(LRL, name='prelu_feature')
        LRL = DropoutLayer(LRL, keep=0.3, is_fix=True, is_train=is_train, name='dropout')
        F_UP = Conv3dLayer(LRL, shape=(3, 3, 3, 64, 256), strides=(1, 1, 1, 1, 1), padding='SAME',
                           W_init=tf.contrib.layers.xavier_initializer(), name='upconv_feature')
        F_UP = DeConv3d(F_UP, 16, filter_size=(3, 3, 3), strides=(1, 4, 4), padding='SAME', name='up_feature')
        F_SR = Conv3dLayer(F_UP, shape=(3, 3, 3, 16, 1), strides=(1, 1, 1, 1, 1), act=tf.nn.leaky_relu, padding='SAME',
                           W_init=tf.contrib.layers.xavier_initializer(), name='sr_feature')

        SR_mapping = []
        frames = UnStackLayer(I_LR, axis=1)
        for frame in frames:
            frame_up = UpSampling2dLayer(frame, (4, 4), is_scale=True, method=0)
            '''
            - Index 0 is ResizeMethod.BILINEAR, Bilinear interpolation.
            - Index 1 is ResizeMethod.NEAREST_NEIGHBOR, Nearest neighbor interpolation.
            - Index 2 is ResizeMethod.BICUBIC, Bicubic interpolation.
            - Index 3 ResizeMethod.AREA, Area interpolation.
            '''
            SR_mapping.append(frame_up)
        frames_up = StackLayer(SR_mapping, axis=1)

        sr = ElementwiseLayer(layers=[F_SR, frames_up], combine_fn=tf.add, name='HRL')

    return sr
