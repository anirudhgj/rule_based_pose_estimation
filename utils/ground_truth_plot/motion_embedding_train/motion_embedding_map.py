
from other_utils import np_utils, sk_utils
import numpy as np
import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
import imageio
from termcolor import colored

import model 
from hyperparams import Hyperparameters
import graph 
from data_loader import Data_loader
import model_componets as comps
from sklearn.decomposition import PCA


from numpy import array
from scipy.linalg import svd


H = Hyperparameters ()

D = Data_loader(H.data_path,H.seq_length,H.batch_size)


os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)




input_ph = tf.placeholder(tf.float32, shape= [None , H.num_joints ,3],name = 'skeleton_input')

global_step = tf.train.get_or_create_global_step()

x_input = input_ph

x_input_view_norm,x_input_local,tr_mats = comps.root_relative_to_local(x_input)

encoder_out = graph.apply_pose_encoder(input_ph)

pose_encoder_params = graph.get_network_params("Encoder_net")

encoder_input = tf.reshape(encoder_out,(-1,H.seq_length,32))

encoder_lstm_out = model.apply_encoder(encoder_input,name ='motion_encoder')

z_state = encoder_lstm_out['z_state']

z_outputs = encoder_lstm_out['z_outputs']



decoder_lstm_out = model.apply_decoder(z_state,z_outputs,name = 'motion_decoder')

motion_recon = decoder_lstm_out['x_recon']

motion_recon_reshaped = tf.reshape(motion_recon,((-1,32)))

pose_recon = graph.apply_pose_decoder(motion_recon_reshaped)#view norm

pose_decoder_params = graph.get_network_params("Decoder_net")

param_lstm_encoder = model.get_network_params('motion_encoder')

param_lstm_decoder = model.get_network_params('motion_decoder')




sess.run(tf.global_variables_initializer())

print colored("loading weights","blue")

tf.train.Saver(pose_encoder_params).restore(sess,'../../../pose_embedding_train/ent44_15j_32/weights/encoder_iter-799001') 
print colored("loaded pose_encoder weights","yellow")

tf.train.Saver(pose_decoder_params).restore(sess,'../../../pose_embedding_train/ent44_15j_32/weights/decoder_iter-799001')
print colored("loaded pose_decoder weights","blue")

tf.train.Saver(param_lstm_encoder).restore(sess,tf.train.latest_checkpoint('../../../motion_embedding_train/ent_44_15j_32/BILSTM_TRAIN/weights/lstm_encoder/'))
print colored("loaded pose lstm_encoder weights","green")

tf.train.Saver(param_lstm_decoder).restore(sess,tf.train.latest_checkpoint('../../../motion_embedding_train/ent_44_15j_32/BILSTM_TRAIN/weights/lstm_decoder/'))
print colored("laoded pose lstm_decoder weights","red")


train_batch = np.asarray(D.get_sequence_batch_train())

train_batch = train_batch[:,0:30]


train_batch = train_batch.reshape((-1,H.num_joints , 3))


print train_batch.reshape(-1,H.seq_length,H.num_joints,3).shape

feed_dict = {x_input : train_batch}

pred_ops = sess.run(z_state,feed_dict=feed_dict)

predictions = pred_ops

predictions.shape



pca = PCA(n_components=2)

principal_components = pca.fit_transform(predictions)

eigenvalues = pca.explained_variance_


mean = np.mean(predictions )

print colored(mean,"yellow")

print colored(eigenvalues,"magenta")