import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from termcolor import colored
import scipy.io as sio
import io


import model_componets as comps
from data_loader import DataLoader
from commons import transform_util as tr_util
from model_componets import *
import vis_image as vis


data_loader = DataLoader()

train_batch_size, test_batch_size = 256, 256
lr_disc, lr_encoder, lr_decoder = 0.000002, 0.000002, 0.000002
disp_step_save, disp_step_valid = 1000, 500

num_batches_train = data_loader.get_num_batches('train', 2 * train_batch_size)
num_batches_test = data_loader.get_num_batches('test', 2 * test_batch_size)

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)

def get_network_params(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

input_pose = tf.placeholder(tf.float32, shape = [None,17,3], name = 'input_pose')

x_real = input_pose

x_view_norm_real, x_local_real, transform_mats = comps.root_relative_to_local(x_real)

encoder_real = EncoderNet(x_local_real)

z_real = encoder_real['z_joints']

decoder_real = DecoderNet(z_real)

x_local_recon = decoder_real['full_body_x']

x_recon = comps.local_to_root_relative(x_local_recon, transform_mats)

param_encoder = get_network_params(scope='Encoder_net')
param_decoder = get_network_params(scope='Decoder_net')


tf.train.Saver(param_decoder).restore(sess,'./pretrained_weights/decoder_iter-849001' )
tf.train.Saver(param_encoder).restore(sess,'./pretrained_weights/encoder_iter-849001' )

train_writer = tf.summary.FileWriter('./logs_verify/train',sess.graph)
test_writer = tf.summary.FileWriter('./logs_verify/test')

num_epochs = 10
epoch, iteration_no = 0, 0

def gen_plot(arr):
    plt.figure()
    plt.plot(arr)
    plt.title("sigma vs components")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

final_loss15 = 0
final_loss17 = 0

while epoch < num_epochs:
    epoch += 1
    print epoch
    data_loader.shuffle_data('train')

    for batch_idx in range(num_batches_train):
        iteration_no += 1
        x_inputs = data_loader.get_train_data_batch(train_batch_size, batch_idx)
        feed_dict  = {input_pose : x_inputs}
        pose_recon = sess.run(x_recon , feed_dict = feed_dict)
        x_inputs_fifteen = x_inputs[:,[0,1,2,3,5,6,7,8,9,10,11,13,14,15],:]
        pose_recon_fifteen = pose_recon[:,[0,1,2,3,5,6,7,8,9,10,11,13,14,15],:]
        # pose_embed = sess.run(z_real,feed_dict)



        if iteration_no % 1 ==0 :

            fifteen_loss = np.mean(np.abs(x_inputs_fifteen - pose_recon_fifteen))
            # fifteen_summary = tf.Summary(value=[tf.Summary.Value(tag="fifteen_joints_loss", simple_value=fifteen_loss),])
            # train_writer.add_summary(fifteen_summary,iteration_no)
            # train_writer.flush()

            final_loss15 = final_loss15 + fifteen_loss

            seventeen_loss = np.mean(np.abs(x_inputs - pose_recon))
            # seventeen_summary = tf.Summary(value=[tf.Summary.Value(tag="seventeen_joints_loss", simple_value=seventeen_loss),])
            # train_writer.add_summary(seventeen_summary,iteration_no)
            # train_writer.flush()

            final_loss17 = final_loss17 + seventeen_loss

            # var_arr = np.var(pose_embed,axis = 0)
            # plot_buf = gen_plot(var_arr)
            # image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            # image = tf.expand_dims(image, 0)
            # summary_op = tf.summary.image("plot", image)
            # summary_image = sess.run(summary_op,feed_dict)
            # train_writer.add_summary(summary_image)
            # train_writer.flush()

            # print fifteen_loss, seventeen_loss
            # print iteration_no

        if iteration_no % 1000 == 0:

            print iteration_no , fifteen_loss , seventeen_loss

print final_loss15/iteration_no
print final_loss17/iteration_no
