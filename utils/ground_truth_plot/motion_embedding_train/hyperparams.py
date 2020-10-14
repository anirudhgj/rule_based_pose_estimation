import numpy as np


class Hyperparameters(object):

    def __init__(self):

        self.exp_name = 'rule_based_motion_net_expt_seq30'
        self.data_path = '../../../data/data_15j/'
        #self.data_path = '../../../../17j/codes_2020/data/vneck/combined/all_data.mat'
        
        self.logdir_path_train = './logs/'+self.exp_name+'/train'
        self.logdir_path_val = './logs/'+self.exp_name+'/val'
        self.videos_path = './videos/'+self.exp_name
        self.store_weights_path = './weights/whole/'
        self.load_weights_path = './weights/whole/'
        self.store_encoder_weights = './weights/lstm_encoder/'
        self.store_decoder_weights = './weights/lstm_decoder/'


        self.num_joints = 15
        #self.state_size = 128
        self.state_size = 32
        
        self.batch_size = 3
        self.max_iterations = 100000
        self.val_after = 100


        self.seq_length = 30


        self.num_stacked_lstm_layers = 2
