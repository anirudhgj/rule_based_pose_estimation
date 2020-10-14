# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 19:08:50 2019

@author: anirudh
"""

import numpy as np
from scipy.io import loadmat,savemat
from hyperparams import Hyperparameters
import os,glob
import random
import time


H = Hyperparameters()


class Data_loader(object):
    def __init__(self,data_path,sequence_length,batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.sequence_length =sequence_length
        self.datasets = os.listdir(self.data_path)
        self.val_list = ['Taichi_S6', 'HipHop_HipHop6', 'Jazz_Jazz6', 'Sports_Tennis_Left']
        self.all_videos = [val for sublist in [[os.path.join(i[0], j) for j in i[2] if j.endswith('.mat')] for i in os.walk(self.data_path+'/train/')] for val in sublist]

    def get_sequence_batch_train(self):

        my_video = np.random.choice(self.all_videos,self.batch_size)
        sequences  = [random.randint(0,loadmat(video)['pose_3d'].shape[0]-self.sequence_length) for video in my_video]
        batch = []
        for i in range(len(sequences)):
            k=loadmat(my_video[i])['pose_3d'][sequences[i]:sequences[i] + H.seq_length].reshape((H.seq_length,H.num_joints * 3))
            batch.append(k)
        return batch
        
    def get_sequence_batch_valid(self):
        
        mads_videos = os.listdir(os.path.join(self.data_path,'valid/'))
        videos = [video for video in mads_videos if video.rsplit('_',1)[0] in self.val_list]
        count = 1
        batch= []
        while True :
            if count > self.batch_size:
                break
            rv=random.choice(videos)
            seq_no = random.randint(0,loadmat(os.path.join(self.data_path,'valid',rv))['pose_3d'].shape[0]-self.sequence_length)
            k=loadmat(os.path.join(self.data_path,'valid',rv))['pose_3d'][seq_no:seq_no + self.sequence_length].reshape((self.sequence_length,H.num_joints*3))
            batch.append(k) 
            count = count + 1
                
        return batch
        

# data_path = '../data/vneck_pose_sequence/'
# sequence_length = 90
# batch_size = 64        
# dataloader = Data_loader(data_path,sequence_length,batch_size)
# start = time.clock()
# k=np.array(dataloader.get_sequence_batch_train())
# print (k.shape)
# print (time.clock() - start)
        

