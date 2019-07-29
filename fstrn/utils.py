#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import h5py
import skimage
import skimage.measure


def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
    epsilon = 1e-6
    if is_mean:
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1, tensor2)) + epsilon), [1, 2, 3]))
    else:
        loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1, tensor2)) + epsilon), [1, 2, 3]))
    return loss



def to_frames(x5):
    size = np.size(x5)
    x4 = []
    for images in x5:
        for image in images:
            x4.append(image)
    return np.array(x4)


def load_data(path):
    f = h5py.File(path)
    lr_data = np.asarray(f.get('lrs_data')[:], dtype=np.float32).T  # lr
    hr_data = np.asarray(f.get('hr_data')[:], dtype=np.float32).T  # hr
    blr_data = np.asarray(f.get('lr_data')[:], dtype=np.float32).T  # bicubic
    return lr_data, hr_data, blr_data


def PSNR(hr_images, sr_images):
    psnrs = []
    for hr_image, sr_image in zip(hr_images, sr_images):
        psnr = skimage.measure.compare_psnr(hr_image, sr_image, data_range=1)
        psnrs.append(psnr)
    return psnrs


def SSIM(hr_images, sr_images):
    ssims = []
    if np.size(hr_images, -1) == 1:
        multichannel = False
    else:
        multichannel = True
    for hr_image, sr_image in zip(hr_images, sr_images):
        if multichannel == False:
            hr_image = np.squeeze(hr_image)
            sr_image = np.squeeze(sr_image)
        ssim = skimage.measure.compare_ssim(hr_image, sr_image, data_range=1, multichannel=multichannel)
        ssims.append(ssim)
    return ssims
