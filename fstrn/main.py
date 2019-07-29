#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, random
import tensorlayer as tl
from model import fstrn
from utils import *
from config import config


###====================== HYPER-PARAMETERS ===========================###
batch_size = config.TRAIN.batch_size
time_step = config.time_step
lr = config.TRAIN.lr
beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch
lr_decrease = config.TRAIN.lr_decrease
checkpoint_dir = "checkpoint"

def train():
    # create folders to trained model
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_imgmat = '25_seq_13020_yuv_scale_4_frm5_blur_2.mat'
    print('[**] LOADING TRAIN SET')
    print('Loading %s...' % train_imgmat)
    train_lr_imgs, train_hr_imgs, _ = load_data(config.TRAIN.path + train_imgmat) # lr_data, hr_data, blr_data
    train_hr_imgs = np.transpose(train_hr_imgs, (0, 1, 3, 4, 2)) # channel last
    train_lr_imgs = np.transpose(train_lr_imgs, (0, 1, 3, 4, 2))

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [batch_size, time_step, 36, 36, 1], name='t_image_input')
    t_target_image = tf.placeholder('float32', [batch_size, time_step, 144, 144, 1], name='t_target_image')

    net_image = fstrn(t_image, is_train=True, reuse=False)
    net_image.print_params(False)


    ###========================== DEFINE TRAIN OPS ==========================###
    loss = compute_charbonnier_loss(net_image.outputs[:, (time_step-1)//2:(time_step-1)//2+1, :, :, :],
                                    t_target_image[:, (time_step-1)//2:(time_step-1)//2+1, :, :, :], is_mean=True) # center frame
    g_vars = tl.layers.get_variables_with_name('FSTRN', True, True)
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False, dtype=tf.float32)
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=g_vars)


    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    ###============================= TRAINING ===============================###
    history_errs = []
    bad_count = 0
    bad_max = 5
    for epoch in range(0, n_epoch + 1):
        if epoch == 0:
            sess.run(tf.assign(lr_v, lr))
            log = " ** init lr: %f  " % (sess.run(lr_v))
            print(log)

        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        random_imgs_index = np.array(range(0, len(train_hr_imgs)))
        random.shuffle(random_imgs_index) # prepare shuffled data every epoch
        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            train_index = random_imgs_index[idx:idx + batch_size]
            b_imgs_240 = train_hr_imgs[train_index]
            b_imgs_60 = train_lr_imgs[train_index]
            if np.size(b_imgs_240, 0) != batch_size or np.size(b_imgs_240, 1) != time_step:
                continue
            errM, _ = sess.run([loss, g_optim_init], {t_image: b_imgs_60, t_target_image: b_imgs_240})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, loss: %.8f " % (epoch, n_epoch, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)
        history_errs.append(total_mse_loss / n_iter)
        if (len(history_errs) > bad_max and total_mse_loss / n_iter < np.array(history_errs)[:-1].min()):
            bad_count = 0
        elif (len(history_errs) > bad_max and total_mse_loss / n_iter >= np.array(history_errs)[-bad_max:-1].min()):
            bad_count += 1
            if bad_count > bad_max:
                new_lr = sess.run(lr_v) * lr_decrease
                sess.run(tf.assign(lr_v, new_lr))
                log = " ** new learning rate: %f (for init)" % (sess.run(lr_v))
                print(log)
                bad_count = 0

        # save model
        if (epoch % 10 == 0):
            npz_name = 'fstrn_{:05d}.npz'.format(epoch)
            tl.files.save_npz(net_image.all_params, name=checkpoint_dir + '/' + npz_name, sess=sess)
            print(npz_name + ' saved')
            # evaluate(npz_name)


def evaluate(npzFile, valid_imgmat):

    print(' [**] LOADING VALID SET')
    print('Loading %s...' % valid_imgmat)
    valid_lr_imgs, valid_hr_imgs, valid_blr_imgs = load_data(config.VALID.path + valid_imgmat)
    print(np.shape(valid_lr_imgs), np.shape(valid_hr_imgs), np.shape(valid_blr_imgs))
    print(valid_hr_imgs.min(), valid_hr_imgs.max())
    valid_lr_imgs = np.transpose(valid_lr_imgs, (0, 1, 3, 4, 2))
    valid_hr_imgs = np.transpose(valid_hr_imgs, (0, 1, 3, 4, 2))
    valid_blr_imgs = np.transpose(valid_blr_imgs, (0, 1, 3, 4, 2))
    print(np.shape(valid_lr_imgs), np.shape(valid_hr_imgs), np.shape(valid_blr_imgs))
    print(valid_hr_imgs.min(), valid_hr_imgs.max())
    valid_lr_imgs = np.insert(valid_lr_imgs, 0, valid_lr_imgs[0][0], axis=1)
    valid_lr_imgs = np.insert(valid_lr_imgs, 0, valid_lr_imgs[0][0], axis=1)  # head padding
    valid_lr_imgs = np.append(valid_lr_imgs, [[valid_lr_imgs[0][-1]]], axis=1)
    valid_lr_imgs = np.append(valid_lr_imgs, [[valid_lr_imgs[0][-1]]], axis=1)  # tail padding
    
    ###========================== DEFINE MODEL ============================###
    size = valid_lr_imgs.shape
    t_image = tf.placeholder('float32', [1, time_step, size[-3], size[-2], size[-1]], name='input_image')

    net_image = fstrn(t_image, is_train=False, reuse=tf.AUTO_REUSE)


    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    tl.layers.initialize_global_variables(sess)

    ###======================= RESTORING G & EVALUATION =============================###
    npzFilename = os.path.splitext(npzFile)[0]
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/' + npzFile, network=net_image)
    start_time = time.time()
    srs = []
    for i in range(0, np.size(valid_hr_imgs, 1), 1):
        b_imgs_60 = valid_lr_imgs[:, i:i + time_step, :, :, :]
        outs = sess.run(net_image.outputs, {t_image: b_imgs_60})
        srs.append(outs[0][2]) # center frame
    b_imgs_240 = to_frames(valid_hr_imgs)
    outs = srs
    out_bicus = to_frames(valid_blr_imgs)

    psnr_gen = PSNR(b_imgs_240, outs)
    psnr_bic = PSNR(b_imgs_240, out_bicus)
    ssim_gen = SSIM(b_imgs_240, outs)
    ssim_bic = SSIM(b_imgs_240, out_bicus)

    print("Using %s on %s took: %4.4fs" % (npzFilename, valid_imgmat[:-4], time.time() - start_time))
    print("[**]PSNR_gen Avg: %f, SSIM_gen Avg: %f" % (np.mean(psnr_gen), np.mean(ssim_gen)))
    print("[**]PSNR_bic Avg: %f, SSIM_bic Avg: %f" % (np.mean(psnr_bic), np.mean(ssim_bic)))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, evaluate')
    parser.add_argument('--npzName', type=str, default='', help='Input the npz file you want to use')
    parser.add_argument('--matName', type=str, default='', help='Input the mat file you want to use')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode
    tl.global_flag['npzName'] = args.npzName
    tl.global_flag['matName'] = args.matName


    if tl.global_flag['mode'] == 'train':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate(tl.global_flag['npzName'], tl.global_flag['matName'])
    else:
        raise Exception("Unknow --mode")
