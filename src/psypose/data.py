#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:50:03 2021

@author: f004swn
"""

import os
import joblib
import numpy as np
import cv2
#import matplotlib
import nibabel as nib
from tqdm import tqdm

import pandas as pd

#os.system('cd ..')
from psypose import utils

import torch

class pose(object):
    
    def __init__(self):
        self.is_clustered = False
        self.clusters_named = False
        self.shots = None
        pass
            
    def load_fmri(self, fmri_path, TR):
        self.fmri_path = fmri_path
        self.brain_data = nib.load(fmri_path).get_fdata()
        # pull TR from file header
        self.TR = TR
        # all timeseries will be in milliseconds for accuracy.
        self.brain_time = [1000*TR for tr in range(self.brain_data.shape[0])]
        
    # def load_face_data(self, face_data):
    #     self.face_data_path = os.path.abspath(face_data)
        
    #     face_data_array = np.genfromtxt(face_data, delimiter=',')
    #     n_frames = face_data_array.shape[0]
    #     face_df = pd.DataFrame(columns=['frame_ids', 'locations', 'encodings'])
    #     face_df['frame_ids'] = face_data_array[:,0]
    #     face_df['locations'] = [face_data_array[i,1:5] for i in range(n_frames)]
    #     face_df['encodings'] = [face_data_array[i,5:] for i in range(n_frames)]
    #     self.face_data = face_df
        
    def load_face_data(self, face_data):
        #expects py-feat face df in csv
        self.face_data_path = os.path.abspath(face_data)
        self.face_data = pd.read_csv(face_data)

    def load_video(self, vid_path):
        vid_path = os.path.abspath(vid_path)
        self.vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        self.video_cv2 = cv2.VideoCapture(vid_path)
        #self.video_scenedetect = VideoManager([vid_path])
        self.fps = self.video_cv2.get(cv2.CAP_PROP_FPS)
        self.vid_path = vid_path
        self.framecount = int(self.video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_time = [(1/self.fps)*1000*frame for frame in range(self.framecount)]
        self.video_shape = (int(self.video_cv2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cv2.get(cv2.CAP_PROP_FRAME_WIDTH)))
        
    def load_pkl(self, pkl_path):
        self.pkl_path = pkl_path
        global pkl_open
        pkl_open = dict(joblib.load(pkl_path))
        self.pose_data = pkl_open
        self.n_tracks = len(pkl_open)

        
    # def consolidate_clusters(self):
    #     tot_frames = int(self.video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
