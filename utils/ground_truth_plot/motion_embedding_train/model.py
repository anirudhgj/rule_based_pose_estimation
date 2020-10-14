import tensorflow as tf
import numpy as np

from hyperparams import Hyperparameters
import utils

H = Hyperparameters() 


def apply_encoder(input_x_seq,name,reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        cell_fw_encoder = tf.nn.rnn_cell.MultiRNNCell(
            [utils.get_a_cell(H.state_size) for _ in range(H.num_stacked_lstm_layers)]
        )

        cell_bw_encoder = tf.nn.rnn_cell.MultiRNNCell(
            [utils.get_a_cell(H.state_size) for _ in range(H.num_stacked_lstm_layers)]
        )

        outputs_encoder, state_encoder = tf.nn.bidirectional_dynamic_rnn(cell_fw_encoder, cell_bw_encoder, input_x_seq, dtype=tf.float32)

        final_state_fw = state_encoder[0][1][1]

        final_state_bw = state_encoder[1][1][1]

        final_state = tf.concat([final_state_fw, final_state_bw], 1)

        act = lambda x: 10 * tf.tanh(x)

        final_state_fced = tf.contrib.layers.fully_connected(final_state, H.state_size, activation_fn=act)

        final_state_fced_stacked = tf.stack([final_state_fced]*input_x_seq.shape[1], 1)

        encoder_net={
            'input' : input_x_seq,
            'z_state':final_state_fced,
            'z_outputs' : final_state_fced_stacked

        }

    return encoder_net

def apply_decoder(z_state,z_outputs,name,reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        final_state_fced = z_state

        final_state_fced_stacked = z_outputs

        rnn_tuple_state_fw = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(final_state_fced, final_state_fced)
            for idx in range(H.num_stacked_lstm_layers)]
            )

        rnn_tuple_state_bw = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(final_state_fced, final_state_fced)
            for idx in range(H.num_stacked_lstm_layers)]
            )

        cell_fw_decoder = tf.nn.rnn_cell.MultiRNNCell(
            [utils.get_a_cell(H.state_size) for _ in range(H.num_stacked_lstm_layers)]
        )

        cell_bw_decoder = tf.nn.rnn_cell.MultiRNNCell(
            [utils.get_a_cell(H.state_size) for _ in range(H.num_stacked_lstm_layers)]
        )

        outputs_decoder, state_decoder = tf.nn.bidirectional_dynamic_rnn(cell_fw_decoder, cell_bw_decoder, 
                                                                    final_state_fced_stacked, 
                                                                    initial_state_fw=rnn_tuple_state_fw, initial_state_bw=rnn_tuple_state_bw)

        outputs_decoder_merged = tf.concat(outputs_decoder, 2)

        final_output_pred = tf.contrib.layers.fully_connected(outputs_decoder_merged, 32, activation_fn=None)


        decoder_net ={
            'z_state' :z_state,
            'z_outputs' : z_outputs,
            'x_recon' : final_output_pred 
        }

        return decoder_net

def fill_net(input,name,reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        out_1 = tf.contrib.layers.fully_connected(input, 512)

        out_2 = tf.contrib.layers.fully_connected(out_1, 256)

        out_3 = tf.contrib.layers.fully_connected(out_2, H.state_size,activation_fn=None)
        
        mapped={
            'mapped_motion':out_3
        }

    return mapped



def get_network_params(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)