import matplotlib
matplotlib.use("Agg")
import traceback
import os, time
import sys
import utils
import numpy as np
import tensorflow as tf
import math as m
# import load_batch_data as data_loader
from data_loader import DataLoader
from commons import transform_util as tr_util
from model_componets import *
import model as M
from termcolor import colored


import matplotlib.pyplot as plt
import vis_image as vis

os.system('rm -rf logs')

data_loader = DataLoader()

cur_dir_name = os.path.basename(os.path.abspath('.'))

if not os.path.exists('./weights/AE_humans_mads_yt_mpi_l1/whole/'):
    os.makedirs('./weights/AE_humans_mads_yt_mpi_l1/whole/')

del_logs_flag = '--del-l' in sys.argv
del_weights_flag = '--del-w' in sys.argv
load_weights_flag = '--load-w' in sys.argv
resume_flag = '--resume' in sys.argv

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

train_batch_size, test_batch_size = 256,256
lr_disc, lr_encoder, lr_decoder = 0.000002, 0.000002, 0.000002
disp_step_save, disp_step_valid = 1000, 500

num_batches_train = data_loader.get_num_batches('train', 2 * train_batch_size)
num_batches_test = data_loader.get_num_batches('test', 2 * test_batch_size)

#################################################################################
############################# Start Session #####################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session_1 = tf.InteractiveSession(config=config)

####################### To load pretrained weights ##############################
session_1.run(tf.global_variables_initializer())




epoch, iteration_no = 0, 0
resume_flag = True

'''
if resume_flag or load_weights_flag:
    os.system('mkdir -p pretrained_weights')
    try:
        # iteration_no = utils.copy_latest('iter')
        num = 237000
        print 'Loading encoder-decoder weights from iter'
        print 'Loading disc weights from iter-%d' % iteration_no
        M.load_weights(num, session_1, dir='./weights/AE_humans_mads_yt_mpi_l1')

        print colored(num,"yellow")
    except:
        traceback.print_exc()
        print ('Some Problem Occured while resuming... starting from iteration {}'.format(iteration_no))
        raise Exception('Ehh! cant load iter: {}'.format(iteration_no))

if del_weights_flag:
    print 'deleting weights...'
    os.system('rm -rf weights/AE/')
'''

saver_best_decoder = tf.train.Saver(M.param_decoder, max_to_keep=5)
saver_best_encoder = tf.train.Saver(M.param_encoder, max_to_keep=5)
saver_best_disc = tf.train.Saver(M.param_disc, max_to_keep=5)

saver_iter_decoder = tf.train.Saver(M.param_decoder, max_to_keep=5)
saver_iter_encoder = tf.train.Saver(M.param_encoder, max_to_keep=5)
saver_iter_disc = tf.train.Saver(M.param_disc, max_to_keep=5)
saver_iter_batch_disc = tf.train.Saver(M.param_disc, max_to_keep=5)
saver = tf.train.Saver()

####################################
##### initialise variables #########
num_epochs = 100000
disc_train_loss_, decoder_train_loss_, encoder_train_loss_ = [0], [0], [0]
gen_adv_train_loss, cyclic_train_loss_, disc_train_acc_, gen_train_acc = [0], [0], [0], [0]

disc_train_acc_real_ , disc_train_acc_fake_ = [0],[0]

disc_val_loss_min_, decoder_val_loss_min_, encoder_val_loss_min_ = [], [], []
disc_val_loss_, decoder_val_loss_, encoder_val_loss_ = [], [], []


disc_val_yt_loss_min_, decoder_val_yt_loss_min_, encoder_val_yt_loss_min_ = [], [], []
disc_val_yt_loss_, decoder_val_yt_loss_, encoder_val_yt_loss_ = [], [], []



### for tensorboard ###
train_writer = tf.summary.FileWriter('./logs/AE_humans_mads_yt_mpi_l1/train', session_1.graph)
test_writer = tf.summary.FileWriter('./logs/AE_humans_mads_yt_mpi_l1/test')

s = time.time()
outputs_disc_train = [M.disc_train_op, M.loss_disc, M.disc_acc,M.disc_acc_fake,M.disc_acc_real, M.summary_merge_all]
outputs_encoder_train = [M.encoder_train_op, M.loss_encoder, M.summary_merge_all]
outputs_decoder_train = [M.decoder_train_op, M.loss_decoder, M.loss_cyclic, M.loss_gen_adv, M.disc_acc, M.disc_acc_fake, M.x_view_norm_real, M.x_recon_resnet ,M.x_view_norm_fake, M.summary_merge_all]

flag_train_disc = True
flag_only_x = True
count_gen_iteration, count_disc_iteration = 0., 0.

weight_joint, weight_joint_group, weight_full = 1. / (3 * 17), 1. / (3 * 5), 1. / (3 * 1)
weight_vec = np.array([weight_joint] * 17 + [weight_joint_group] * 5 + [weight_full])
weight_per_joint = np.array([ 0.        ,  3.35804649,  4.54366344,  6.33468266,  9.48821374,\
        4.34761218,  6.78837365, 10.33182082,  5.50529409,  2.08730905,\
        5.77913439,  8.19677145,   1.81806272,  5.34718556,\
        7.78632864])
start_time = time.time()

fig=vis.get_figure()
vis_placeholder1 = tf.placeholder(tf.uint8, vis.fig2rgb_array(fig).shape)
vis_summary1     = tf.summary.image('x-recon', vis_placeholder1)

fig__=vis.get_figure(figsiz=(16,8))
vis_placeholder2 = tf.placeholder(tf.uint8, vis.fig2rgb_array(fig__).shape)
vis_summary2     = tf.summary.image('x-rand', vis_placeholder2)

vis_placeholder3 = tf.placeholder(tf.uint8, vis.fig2rgb_array(fig).shape)
vis_summary3     = tf.summary.image('x-flip', vis_placeholder3)

vis_placeholder4 = tf.placeholder(tf.uint8, vis.fig2rgb_array(fig).shape)
vis_summary4     = tf.summary.image('x-recon_youtube', vis_placeholder4)

summaries = [vis_summary1,  vis_summary2, vis_summary3, vis_summary4]
placeholders = [vis_placeholder1, vis_placeholder2, vis_placeholder3,vis_placeholder4]

M.load_weights(1475001, session_1)

def visualizer(x_recon,  poses, global_step,  train_writer, vis_summary, vis_placeholder, fs=(8,8)):
    fig = vis.get_figure(figsiz=fs)
    plot_buf = vis.gen_plot_2(fig, x_recon, poses )
    vis.figure_to_summary(fig, global_step,  train_writer, vis_summary, vis_placeholder)


# left_indices = [2, 3, 4, 9, 10, 11, 12]
# right_indices = [5, 6, 7, 13, 14, 15, 16]


# def x_flip(sk_17):
#     sk_17_flip = sk_17.copy()
#     l_idx, r_idx = left_indices, right_indices
#     sk_17_flip[:, l_idx] = sk_17[:, r_idx]
#     sk_17_flip[:, r_idx] = sk_17[:, l_idx]
#     sk_17_flip[:, :, 0] *= -1
#     sk_17 = sk_17_flip
#     return sk_17



# def augment_pose_seq(pose_seq,z_limit=(0,360),y_limit=(-90,90)):
#     pose_seq = np.expand_dims(pose_seq, axis=1)
#     thetas = np.random.uniform(z_limit[0],z_limit[1], pose_seq.shape[0])
#     thetas = np.stack([thetas]*pose_seq.shape[1], 1)
#     k=[]
#     for ct, xx in enumerate(thetas):
#         k.append(pose_rotate(pose_seq[ct], np.expand_dims(thetas[ct], 1), pose_seq[ct].shape[0]))
#     k = np.stack(k, 0)

#     thetas = np.random.uniform(y_limit[0],y_limit[1], k.shape[0])
#     thetas = np.stack([thetas]*k.shape[1], 1)
#     p=[]
#     for ct, xx in enumerate(thetas):
#         p.append(rotate_y_axis(k[ct], np.expand_dims(thetas[ct], 1), k[ct].shape[0]))
#     p = np.stack(p, 0)
#     return k

# def pose_rotate(points, theta, batch_size):
#     theta = theta * np.pi / 180.0
#     cos_vals = np.cos(theta)
#     sin_vals = np.sin(theta)
#     row_1 = np.concatenate([cos_vals, -sin_vals], axis=1)# 90 x 2
#     row_2 = np.concatenate([sin_vals, cos_vals], axis=1)# 90 x 2
#     row_12 = np.stack((row_1, row_2), axis=1)#90 x 2 x 2
#     zero_size_row1x2 = np.zeros([batch_size, 1, 2])#90 x 1 x 2
#     r1x2xZero = np.concatenate([row_12, zero_size_row1x2], axis=1)
#     stacker = np.array([0.0, 0.0, 1.0])
#     third_cols = np.reshape(np.tile(stacker, batch_size), [batch_size, 3])
#     third_cols = np.expand_dims(third_cols, 2)
#     rotation_matrix = np.concatenate([r1x2xZero, third_cols], axis=2)
#     return np.matmul(points.reshape([points.shape[0], 17, 3]), rotation_matrix)


# def rotate_y_axis(points,theta,batch_size):
#     theta = theta * np.pi / 180
#     cos_vals = np.cos(theta)#90 x 1
#     sin_vals = np.sin(theta)
#     zero_vals = np.zeros((batch_size,1))
#     ones_vals = np.ones((batch_size,1))
#     row_1 = np.concatenate([cos_vals, zero_vals],axis =1)#90 x2
#     row_2 = np.concatenate([zero_vals ,ones_vals],axis=1)# 90 x 2
#     row_12 = np.stack((row_1, row_2), axis=1)#90 x 2 x 2
#     temp_3 = np.stack((-sin_vals,zero_vals),axis =2)#90 x 1 x 2
#     temp_32 = np.concatenate([row_12,temp_3],axis = 1)#90 x 3 x 2
#     third_cols = np.concatenate([sin_vals,zero_vals,cos_vals],axis=1)#90 x 3
#     third_cols = np.expand_dims(third_cols, 2)
#     rotation_matrix = np.concatenate([temp_32, third_cols], axis=2)
#     return np.matmul(points.reshape([points.shape[0], 17, 3]), rotation_matrix)


#############################################

def pose_rotate(points, theta, batch_size):
    theta = theta * np.pi / 180.0
    cos_vals = np.cos(theta)
    sin_vals = np.sin(theta)
    row_1 = np.concatenate([cos_vals, -sin_vals], axis=1)# 90 x 2
    row_2 = np.concatenate([sin_vals, cos_vals], axis=1)# 90 x 2
    row_12 = np.stack((row_1, row_2), axis=1)#90 x 2 x 2
    zero_size_row1x2 = np.zeros([batch_size, 1, 2])#90 x 1 x 2
    r1x2xZero = np.concatenate([row_12, zero_size_row1x2], axis=1)
    stacker = np.array([0.0, 0.0, 1.0])
    third_cols = np.reshape(np.tile(stacker, batch_size), [batch_size, 3])
    third_cols = np.expand_dims(third_cols, 2)
    rotation_matrix = np.concatenate([r1x2xZero, third_cols], axis=2)
    return np.matmul(points.reshape([points.shape[0], 17, 3]), rotation_matrix)


def rotate_y_axis(points,theta,batch_size):
    theta = theta * np.pi / 180
    cos_vals = np.cos(theta)#90 x 1
    sin_vals = np.sin(theta)
    zero_vals = np.zeros((batch_size,1))
    ones_vals = np.ones((batch_size,1))
    row_1 = np.concatenate([cos_vals, zero_vals],axis =1)#90 x2
    row_2 = np.concatenate([zero_vals ,ones_vals],axis=1)# 90 x 2
    row_12 = np.stack((row_1, row_2), axis=1)#90 x 2 x 2
    temp_3 = np.stack((-sin_vals,zero_vals),axis =2)#90 x 1 x 2
    temp_32 = np.concatenate([row_12,temp_3],axis = 1)#90 x 3 x 2
    third_cols = np.concatenate([sin_vals,zero_vals,cos_vals],axis=1)#90 x 3
    third_cols = np.expand_dims(third_cols, 2)
    rotation_matrix = np.concatenate([temp_32, third_cols], axis=2)
    return np.matmul(points.reshape([points.shape[0], 17, 3]), rotation_matrix)


def augment_pose_seq(pose_seq,z_limit=(0,360),y_limit=(-90,90)):
    pose_seq = np.expand_dims(pose_seq, axis=1)
    thetas = np.random.uniform(z_limit[0],z_limit[1], pose_seq.shape[0])
    thetas = np.stack([thetas]*pose_seq.shape[1], 1)
    k=[]
    for ct, xx in enumerate(thetas):
        k.append(pose_rotate(pose_seq[ct], np.expand_dims(thetas[ct], 1), pose_seq[ct].shape[0]))
    k = np.stack(k, 0)

    thetas = np.random.uniform(y_limit[0],y_limit[1], k.shape[0])
    thetas = np.stack([thetas]*k.shape[1], 1)
    p=[]
    for ct, xx in enumerate(thetas):
        p.append(rotate_y_axis(k[ct], np.expand_dims(thetas[ct], 1), k[ct].shape[0]))
    p = np.stack(p, 0)
    return k


def x_flip(sk_17,neg):
    left_indices = [2, 3, 4, 9, 10, 11, 12]
    right_indices = [5, 6, 7, 13, 14, 15, 16]
    sk_17_flip = sk_17.copy()
    l_idx, r_idx = left_indices, right_indices
    sk_17_flip[:, l_idx] = sk_17[:, r_idx]
    sk_17_flip[:, r_idx] = sk_17[:, l_idx]
    sk_17_flip[:, :, 0] *= neg
    sk_17 = sk_17_flip
    return sk_17




################################################################################

print colored('Loaded weights')

while epoch < num_epochs:
    epoch += 1
    data_loader.shuffle_data('train')
    for batch_idx in range(num_batches_train):

        iter_start_time = tic = time.time()

        iteration_no += 1

        x_inputs = data_loader.get_train_data_batch(train_batch_size, batch_idx)

        x_batch = np.expand_dims(x_inputs,axis = 1)

        x_batch_aug = np.squeeze(augment_pose_seq(x_batch),axis = 1)

        x_inputs = x_batch_aug.reshape((-1,17,3))


        if np.random.rand() > .5:

            x_inputs_flip = x_flip(x_inputs,1)
            x_inputs_flip = np.squeeze(augment_pose_seq(x_inputs_flip , z_limit=(180,180),y_limit=(0,0)))
            x_inputs_flip = x_flip(x_inputs_flip,-1)
            x_inputs = x_inputs_flip
            
            
        time_after_load = time.time()

        z_j_batch = np.random.uniform(-10, 10, size=[train_batch_size, num_zdim])


        z_j_real = np.random.randn(train_batch_size, num_zdim)

        tac = time.time()

        feed_dict = {
            M.input_x: x_inputs,
            M.input_z: z_j_batch,
            M.weight_vec_ph: weight_vec,
            M.lr_disc_ph: lr_disc,
            M.lr_encoder_ph: lr_encoder,
            M.lr_decoder_ph: lr_decoder,

        }

        if flag_only_x == True:

            outputs = [M.train_only_x_op,M.x_recon_loss_l1,M.x_recon,M.summary_merge_all]

            op_onlyx = session_1.run(outputs, feed_dict)

            time_after_pass = time.time()

            train_writer.add_summary(op_onlyx[-1], global_step=iteration_no)

            time_before_pass = time.time()
            op_disc = session_1.run(outputs_disc_train, feed_dict)

            time_after_pass = time.time()

            train_writer.add_summary(op_disc[-1], global_step=iteration_no)
            disc_train_loss_.append(op_disc[1])
            disc_train_acc_.append(op_disc[2])
            disc_train_acc_fake_.append(op_disc[3])
            disc_train_acc_real_.append(op_disc[4])


            if iteration_no  % 100 == 0 :

                print "iteration_no",iteration_no , "loss : ",op_onlyx[1]

                print cur_dir_name, 'DISC: Minibatch loss at iteration %d of epoch %d: %.5f with accuracy %.3f ' % (
                    iteration_no, epoch, disc_train_loss_[-1], disc_train_acc_[-1])

            if iteration_no %100 == 0:
                visualizer(op_onlyx[2][0], x_inputs[0], iteration_no,   train_writer, summaries[0] ,    placeholders[0])
                print colored("train plots {}".format(iteration_no), "red" )

            if op_onlyx[1] < 0.05 :

                flag_only_x = False

            if iteration_no % 1000 == 0:

                saver.save(session_1, './weights/AE_humans_mads_yt_mpi_l1/whole/whole', global_step=iteration_no)



        ######################################################################
        ############# Whether to train discriminator or genrator ##############

        else :
            # if False:

            if flag_train_disc:#TODOchanged -  original flag_train_disc
                ######## train the discriminator  ############
                count_disc_iteration += 1

                time_before_pass = time.time()
                op_disc = session_1.run(outputs_disc_train, feed_dict)
                time_after_pass = time.time()

                train_writer.add_summary(op_disc[-1], global_step=iteration_no)
                disc_train_loss_.append(op_disc[1])
                disc_train_acc_.append(op_disc[2])
                disc_train_acc_fake_.append(op_disc[3])
                disc_train_acc_real_.append(op_disc[4])
            else:
                ########## train the Encoder  ##############
                count_gen_iteration += 1
                time_before_pass = time.time()
                op_encoder = session_1.run(outputs_encoder_train, feed_dict)
                time_after_pass = time.time()
                encoder_train_loss_.append(op_encoder[1])

                ########## train the Decoder  ##############
                op_decoder = session_1.run(outputs_decoder_train, feed_dict)
                train_writer.add_summary(op_decoder[-1], global_step=iteration_no)

                decoder_train_loss_.append(op_decoder[1])
                cyclic_train_loss_.append(op_decoder[2])
                gen_adv_train_loss.append(op_decoder[3])
                gen_train_acc.append(100 - op_decoder[5])
                if batch_idx %17==0:
                    print("***viz***")
                    visualizer(op_decoder[-3][0], op_decoder[-4][0], iteration_no,   train_writer, summaries[0] ,    placeholders[0])
                    visualizer(op_decoder[-2][0], op_decoder[-2][0], iteration_no,   train_writer, summaries[1] ,    placeholders[1], fs=(16,8))
                    # visualizer(x_inputs_rotate[0], original_inputs[0], iteration_no,   train_writer, summaries[1] ,    placeholders[1])
    #                 visualizer(x_inputs_flip[56], original_inputs[56], iteration_no,   train_writer, summaries[2] ,    placeholders[2])
                    print colored("train plots {}".format(iteration_no), "red" )

            time_after_iter = time.time()

            if (iteration_no - 1) % 20 == 0:
                print 'Current Network: ', ('Dicriminator' if flag_train_disc else 'Generator')
                print '%d s: Starting Iteration %d' % (tic - start_time, iteration_no)
                print 'Time for data_population : %.2f sec' % (tac - iter_start_time)
                print 'Time for network pass    : %.2f sec' % (time_after_pass - time_before_pass)
                print 'Total time for training  : %.2f sec' % (time_after_iter - iter_start_time)
                print cur_dir_name, 'DISC: Minibatch loss at iteration %d of epoch %d: %.5f with accuracy %.3f ' % (
                    iteration_no, epoch, disc_train_loss_[-1], disc_train_acc_[-1])
                print cur_dir_name, 'ENCO: Minibatch loss at iteration %d of epoch %d cyclic: %.5f' % (
                    iteration_no, epoch, encoder_train_loss_[-1])
                print cur_dir_name, 'DECO: Minibatch loss at iteration %d of epoch %d cyclic: %.5f + adv: %.5f = %.5f with acc %.3f' % (
                    iteration_no, epoch, cyclic_train_loss_[-1], gen_adv_train_loss[-1], decoder_train_loss_[-1], gen_train_acc[-1])

            ############################################################################
            ############# Switching Generator to Discriminator training rule ###########
            # '''
            if flag_train_disc == True and (
                    disc_train_acc_fake_[-3] >= 65 and disc_train_acc_real_[-3] >= 80):  ### switching from Disc to generator#changed disc_train_acc from 90 to 1
                print 'Flipping to Generator...'
                flag_train_disc = False
                count_gen_iteration, count_disc_iteration = 0., 0.
            elif flag_train_disc == False and (
                    gen_train_acc[-1] >= 60):  ### switching from Generator to Disc
                print 'Flipping to Discriminator...'
                flag_train_disc = True
                count_gen_iteration, count_disc_iteration = 0., 0.
            # '''
            ########## update the txt files at each 50 iteration #######
            if ((iteration_no - 1) % disp_step_save) == 0:
                print 'DISC: %.2fs Minibatch loss at iteration %d of epoch %d' % ((time.time() - s), iteration_no, epoch)
                print 'SAVE: Iteration %d' % iteration_no
                # : %.5f with accuracy %.3f ' % (#np.mean(loss_disc[-disp_step:]), np.mean(loss_disc_acc[-disp_step:]))
                saver_iter_decoder.save(session_1, 'weights/AE_humans_mads_yt_mpi_l1/decoder_iter', global_step=iteration_no)
                saver_iter_encoder.save(session_1, 'weights/AE_humans_mads_yt_mpi_l1/encoder_iter', global_step=iteration_no)
                saver_iter_disc.save(session_1, 'weights/AE_humans_mads_yt_mpi_l1/disc_iter', global_step=iteration_no)

                saver_iter_disc.save(session_1, 'weights/AE_humans_mads_yt_mpi_l1/BatchDisc_iter', global_step=iteration_no)



            ####################################################################################
            ################### Code for validation of gan architecture ########################
            ####################################################################################
            if (iteration_no - 1) % disp_step_valid == 0:
                print 'Validation for mads...'
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
                    z_j_batch = np.random.uniform(-10, 10, size=[test_batch_size, num_zdim])
                    z_j_real = np.random.randn(train_batch_size, num_zdim)
                    feed_dict = {
                        M.input_x: x_inputs,
                        M.input_z: z_j_batch,
                        M.weight_vec_ph: weight_vec,

                    }
                    op_val = session_1.run([M.loss_encoder, M.x_recon, M.summary_merge_valid], feed_dict)
                    batch_encoder_val_loss[batch_idx_test] = op_val[0]
                    if batch_idx_test == 0:
                        test_writer.add_summary(op_val[-1], global_step=iteration_no)
                        visualizer(op_val[-2][0], x_inputs[0], iteration_no,   test_writer, summaries[0] ,    placeholders[0])

                        print colored("val plots for mads{}".format(iteration_no), "red" )

                time_after_val = time.time()

                encoder_val_loss = batch_encoder_val_loss.mean()
                print "Time taken for validation      : %.2f sec" % (time_after_val - time_before_val)
                print "Validation Losses: Encoder Loss: %.3f " % (encoder_val_loss)
                encoder_val_loss_.append(encoder_val_loss)
                if not encoder_val_loss_min_ or encoder_val_loss < encoder_val_loss_min_[-1]:
                    encoder_val_loss_min_.append(encoder_val_loss)
                    print('Saving best iteration number:', iteration_no)
                    saver_best_encoder.save(session_1, 'weights/AE_humans_mads_yt_mpi_l1/encoder_best', global_step=iteration_no)
                    saver_best_decoder.save(session_1, 'weights/AE_humans_mads_yt_mpi_l1/decoder_best', global_step=iteration_no)
                    saver_best_disc.save(session_1, 'weights/AE_humans_mads_yt_mpi_l1/disc_best', global_step=iteration_no)
