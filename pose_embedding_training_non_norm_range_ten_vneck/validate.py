import traceback

import os, time
import sys

import utils
import numpy as np
import tensorflow as tf

# import load_batch_data as data_loader
from data_loader import DataLoader
from commons import transform_util as tr_util
from model_componets import *
import model as M
from termcolor import colored

import matplotlib
matplotlib.use("Agg")


import matplotlib.pyplot as plt

import vis_image as vis


data_loader = DataLoader()

cur_dir_name = os.path.basename(os.path.abspath('.'))

del_logs_flag = '--del-l' in sys.argv
del_weights_flag = '--del-w' in sys.argv
load_weights_flag = '--load-w' in sys.argv
resume_flag = '--resume' in sys.argv

if del_logs_flag or not resume_flag:
    print '\ndeleting logs/AE_validation\n'
    os.system('rm -rf logs/AE_validation')

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

train_batch_size, test_batch_size = 256, 128

lr_disc, lr_encoder, lr_decoder = 0.000002, 0.000002, 0.000002

disp_step_save, disp_step_valid = 1000, 500

num_batches_train = data_loader.get_num_batches('train', 2 * train_batch_size)
num_batches_test = data_loader.get_num_batches('val', 2 * test_batch_size)

#################################################################################
############################# Start Session #####################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session_1 = tf.InteractiveSession(config=config)

####################### To load pretrained weights ##############################
session_1.run(tf.global_variables_initializer())
epoch, iteration_no = 0, 0

if resume_flag or load_weights_flag:
    os.system('mkdir -p pretrained_weights')
    try:
        iteration_no = utils.copy_latest('iter')
        print 'Loading encoder-decoder weights from iter-%d' % iteration_no
        print 'Loading disc weights from iter-%d' % iteration_no
        M.load_weights(iteration_no, session_1)
    except:
        traceback.print_exc()
        print ('Some Problem Occured while resuming... starting from iteration {}'.format(iteration_no))
        raise Exception('Ehh! cant load iter: {}'.format(iteration_no))

# if del_weights_flag:
#     print 'deleting weights...'
#     os.system('rm -rf weights/AE/')

saver_best_decoder = tf.train.Saver(M.param_decoder, max_to_keep=5)
saver_best_encoder = tf.train.Saver(M.param_encoder, max_to_keep=5)
saver_best_disc = tf.train.Saver(M.param_disc, max_to_keep=5)

saver_iter_decoder = tf.train.Saver(M.param_decoder, max_to_keep=5)
saver_iter_encoder = tf.train.Saver(M.param_encoder, max_to_keep=5)
saver_iter_disc = tf.train.Saver(M.param_disc, max_to_keep=5)
M.load_weights(205001, session_1)

####################################
##### initialise variables #########
num_epochs = 100000
disc_train_loss_, decoder_train_loss_, encoder_train_loss_ = [0], [0], [0]
gen_adv_train_loss, cyclic_train_loss_, disc_train_acc_, gen_train_acc = [0], [0], [0], [0]

disc_val_loss_min_, decoder_val_loss_min_, encoder_val_loss_min_ = [], [], []
disc_val_loss_, decoder_val_loss_, encoder_val_loss_ = [], [], []

### for tensorboard ###
train_writer = tf.summary.FileWriter('./logs/AE_validation/train', session_1.graph)
test_writer = tf.summary.FileWriter('./logs/AE_validation/test')

s = time.time()
outputs_disc_train = [M.disc_train_op, M.loss_disc, M.disc_acc, M.summary_merge_all]
outputs_encoder_train = [M.encoder_train_op, M.loss_encoder, M.summary_merge_all]
outputs_decoder_train = [M.decoder_train_op, M.loss_decoder, M.loss_cyclic, M.loss_gen_adv, M.disc_acc, M.summary_merge_all]

outputs_decoder_train = [M.decoder_train_op, M.loss_decoder, M.loss_cyclic, M.loss_gen_adv, M.disc_acc, M.x_recon, M.summary_merge_all]


outputs_disc_val = [M.loss_disc, M.disc_acc]
outputs_encoder_val = [M.loss_encoder]
outputs_decoder_val = [M.loss_decoder, M.loss_cyclic, M.loss_gen_adv]

flag_train_disc = True
count_gen_iteration, count_disc_iteration = 0., 0.

weight_joint, weight_joint_group, weight_full = 1. / (3 * 17), 1. / (3 * 5), 1. / (3 * 1)
weight_vec = np.array([weight_joint] * 17 + [weight_joint_group] * 5 + [weight_full])
start_time = time.time()

fig=vis.get_figure()
vis_placeholder1 = tf.placeholder(tf.uint8, vis.fig2rgb_array(fig).shape)
vis_summary1     = tf.summary.image('x-recon', vis_placeholder1)

vis_placeholder2 = tf.placeholder(tf.uint8, vis.fig2rgb_array(fig).shape)
vis_summary2 = tf.summary.image('x-recon_test', vis_placeholder2)


vis_placeholder1 = tf.placeholder(tf.uint8, vis.fig2rgb_array(fig).shape)
vis_summary1     = tf.summary.image('x-recon', vis_placeholder1)




summaries = [vis_summary1, vis_summary2]
placeholders = [vis_placeholder1, vis_placeholder2]


def visualizer(x_recon,  poses, global_step,  train_writer, vis_summary, vis_placeholder):

  fig = vis.get_figure()
  plot_buf = vis.gen_plot_2(fig, x_recon[0], poses[0] )
  vis.figure_to_summary(fig, global_step,  train_writer, vis_summary, vis_placeholder)
              

while epoch < num_epochs:
    epoch += 1
    data_loader.shuffle_data('train')
    for batch_idx in range(num_batches_train):

        iter_start_time = tic = time.time()

        iteration_no += 1
        # x_inputs = data_loader.get_train_data_batch(train_batch_size, batch_idx)

        # time_after_load = time.time()

        # z_j_batch = np.random.uniform(-1, 1, size=[train_batch_size, num_zdim])

        # tac = time.time()

        # feed_dict = {
        #     M.input_x: x_inputs,
        #     M.input_z: z_j_batch,
        #     M.weight_vec_ph: weight_vec,
        #     M.lr_disc_ph: lr_disc,
        #     M.lr_encoder_ph: lr_encoder,
        #     M.lr_decoder_ph: lr_decoder,
        # }

        # ######################################################################
        # ############# Whether to train discriminator or genrator ##############
        # if flag_train_disc:
        #     ######## train the discriminator  ############
        #     count_disc_iteration += 1

        #     time_before_pass = time.time()
        #     op_disc = session_1.run(outputs_disc_train, feed_dict)
        #     time_after_pass = time.time()

        #     train_writer.add_summary(op_disc[-1], global_step=iteration_no)
        #     disc_train_loss_.append(op_disc[1])
        #     disc_train_acc_.append(op_disc[2])
        # else:
        #     ########## train the Encoder  ##############
        #     count_gen_iteration += 1
        #     time_before_pass = time.time()
        #     op_encoder = session_1.run(outputs_encoder_train, feed_dict)
        #     time_after_pass = time.time()
        #     # train_writer.add_summary(op_encoder[-1], global_step=iteration_no)
        #     encoder_train_loss_.append(op_encoder[1])

        #     ########## train the Decoder  ##############
        #     op_decoder = session_1.run(outputs_decoder_train, feed_dict)
        #     train_writer.add_summary(op_decoder[-1], global_step=iteration_no)
        #     decoder_train_loss_.append(op_decoder[1])
        #     cyclic_train_loss_.append(op_decoder[2])
        #     gen_adv_train_loss.append(op_decoder[3])
        #     gen_train_acc.append(100 - op_decoder[4])

        #     if batch_idx %10==0:
        #         visualizer(op_decoder[-2], x_inputs, iteration_no,   train_writer, summaries[0] ,    placeholders[0])
        #         print colored("train plots {}".format(iteration_no), "red" )

        # time_after_iter = time.time()

        # if (iteration_no - 1) % 20 == 0:
        #     print 'Current Network: ', ('Dicriminator' if flag_train_disc else 'Generator')
        #     print '%d s: Starting Iteration %d' % (tic - start_time, iteration_no)
        #     print 'Time for data_population : %.2f sec' % (tac - iter_start_time)
        #     print 'Time for network pass    : %.2f sec' % (time_after_pass - time_before_pass)
        #     print 'Total time for training  : %.2f sec' % (time_after_iter - iter_start_time)
        #     print cur_dir_name, 'DISC: Minibatch loss at iteration %d of epoch %d: %.5f with accuracy %.3f ' % (
        #         iteration_no, epoch, disc_train_loss_[-1], disc_train_acc_[-1])
        #     print cur_dir_name, 'ENCO: Minibatch loss at iteration %d of epoch %d cyclic: %.5f' % (
        #         iteration_no, epoch, encoder_train_loss_[-1])
        #     print cur_dir_name, 'DECO: Minibatch loss at iteration %d of epoch %d cyclic: %.5f + adv: %.5f = %.5f with acc %.3f' % (
        #         iteration_no, epoch, cyclic_train_loss_[-1], gen_adv_train_loss[-1], decoder_train_loss_[-1], gen_train_acc[-1])

        # ############################################################################
        # ############# Switching Generator to Discriminator training rule ###########
        # # '''
        # if flag_train_disc == True and (
        #         disc_train_acc_[-1] >= 90 or count_disc_iteration >= 5):  ### switching from Disc to generator
        #     print 'Flipping to Generator...'
        #     flag_train_disc = False
        #     count_gen_iteration, count_disc_iteration = 0., 0.
        # elif flag_train_disc == False and (
        #         gen_train_acc[-1] >= 80 or count_gen_iteration >= 20):  ### switching from Generator to Disc
        #     print 'Flipping to Discriminator...'
        #     flag_train_disc = True
        #     count_gen_iteration, count_disc_iteration = 0., 0.
        # # '''
        # ########## update the txt files at each 50 iteration #######
        # # if ((iteration_no - 1) % disp_step_save) == 0:
        # #     print 'DISC: %.2fs Minibatch loss at iteration %d of epoch %d' % ((time.time() - s), iteration_no, epoch)
        # #     print 'SAVE: Iteration %d' % iteration_no
        # #     #: %.5f with accuracy %.3f ' % (#np.mean(loss_disc[-disp_step:]), np.mean(loss_disc_acc[-disp_step:]))
        # #     saver_iter_decoder.save(session_1, 'weights/AE/decoder_iter', global_step=iteration_no)
        # #     saver_iter_encoder.save(session_1, 'weights/AE/encoder_iter', global_step=iteration_no)
        # #     saver_iter_disc.save(session_1, 'weights/AE/disc_iter', global_step=iteration_no)



        ####################################################################################
        ################### Code for validation of gan architecture ########################
        ####################################################################################
        if (iteration_no - 1) % disp_step_valid == 0:
            print 'Validation...'
            time_before_val = time.time()
            max_test_batch_iterations = 10
            num_test_batch_iterations = min(num_batches_test, max_test_batch_iterations)

            batch_encoder_val_loss = np.zeros((num_test_batch_iterations,))
            batch_decoder_val_loss = np.zeros((num_test_batch_iterations,))
            batch_disc_val_loss = np.zeros((num_test_batch_iterations,))
            batch_disc_val_acc = np.zeros((num_test_batch_iterations,))

            data_loader.shuffle_data('test')
            for batch_idx_test in range(num_test_batch_iterations):
                x_inputs = data_loader.get_test_data_batch(test_batch_size, batch_idx_test)
                z_j_batch = np.random.uniform(-1, 1, size=[test_batch_size, num_zdim])
                feed_dict = {
                    M.input_x: x_inputs,
                    M.input_z: z_j_batch,
                    M.weight_vec_ph: weight_vec,
                }
                op_val = session_1.run([M.loss_encoder, M.x_recon, M.summary_merge_valid], feed_dict)
                batch_encoder_val_loss[batch_idx_test] = op_val[0]
                if batch_idx_test == 0:
                    test_writer.add_summary(op_val[-1], global_step=iteration_no)
                    visualizer(op_val[-2], x_inputs, iteration_no,   test_writer, summaries[1] ,    placeholders[1])
                    print colored("val plots {}".format(iteration_no), "red" )

            time_after_val = time.time()

            encoder_val_loss = batch_encoder_val_loss.mean()
            print "Time taken for validation      : %.2f sec" % (time_after_val - time_before_val)
            print "Validation Losses: Encoder Loss: %.3f " % (encoder_val_loss)
            ########### Save best validation  ##################
            encoder_val_loss_.append(encoder_val_loss)
            if not encoder_val_loss_min_ or encoder_val_loss < encoder_val_loss_min_[-1]:
                encoder_val_loss_min_.append(encoder_val_loss)
                print('Saving best iteration number:', iteration_no)
