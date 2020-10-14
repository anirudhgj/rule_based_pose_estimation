import numpy as np


class Hyperparameters(object):

    def __init__(self):

        self.exp_name = 'motion_net_expt_seq30_HuMaMpi'
        self.data_path = '../data/vneck_pose_sequence/'
        self.rule_network_name = 'FlippedForwardNormalBackward'
        self.logdir_path_train = './logs/'+self.exp_name+'/train'
        self.logdir_path_val = './logs/'+self.exp_name+'/val'
        self.videos_path = './videos/'+self.exp_name
        self.store_weights_path = './weights/whole/'
        self.load_weights_path = './weights/whole/'
        self.store_encoder_weights = './weights/lstm_encoder/'
        self.store_decoder_weights = './weights/lstm_decoder/'
        self.store_rule_network_weights = './weights/'+self.rule_network_name+'/'

        self.pose_embedding_size = 32
        self.num_joints = 17
        self.state_size = 128
        self.batch_size = 64
        self.max_iterations = 100000
        self.val_after = 100
        self.seq_length = 30
        self.num_stacked_lstm_layers = 2
