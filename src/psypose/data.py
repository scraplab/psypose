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
import nibabel as nib
from tqdm import tqdm
from quaternion import as_quat_array
import warnings

import pandas as pd

#os.system('cd ..')
from psypose import utils

import torch


def check_keys(obj, keystr):
    allkeys = list(obj.keys())
    if keystr in allkeys:
        return True
    else:
        return False

class pose(object):
    
    def __init__(self):
        self.is_clustered = False
        self.clusters_named = False
        self.shots = None
        self.is_raw = False
        self.face_data = None
        self.face_data_path = None
        self.pose_data = None
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
        #self.video_array = utils.video_to_array(self.video_cv2)
        #self.video_bytes = utils.video_to_bytes(self.video_cv2)
        #del video_array
        
    def load_pkl(self, pkl_path):
        self.pkl_path = pkl_path
        global pkl_open
        pkl_open = joblib.load(pkl_path)
        if isinstance(pkl_open, list):
            self.is_raw=True
        # else:
        #     for track_id, data in pkl_open.items():
        #         data['quaternion'] = as_quat_array(data['quaternion'])
        self.n_tracks = len(pkl_open)
        self.pose_data = pkl_open

    def reinit_video(self):
        del self.video_cv2
        self.video_cv2 = cv2.VideoCapture(self.vid_path)
        
    # def consolidate_clusters(self):
    #     tot_frames = int(self.video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))

    def save_to_pose(self, save_as = None):
        out_obj = {}
        out_obj['vid_path'] = self.vid_path
        out_obj['vid_name'] = self.vid_name
        out_obj['framecount'] = self.framecount
        out_obj['fps'] = self.fps
        out_obj['video_time'] = self.video_time
        out_obj['video_shape'] = self.video_shape
        if self.face_data is not None:
            out_obj['face_data'] = self.face_data
            if self.face_data_path is not None:
                out_obj['face_data_path'] = self.face_data_path
        if self.pose_data is not None:
            out_obj['pose_data'] = self.pose_data

        if not save_as:
            save_as = os.path.join(os.getcwd(), self.vid_name) + '.pose'

        print('Writing to pose_object...')
        joblib.dump(out_obj, save_as)

    def load_pose(self, pose_path):
        if 'pkl' in pose_path:
            warnings.Warn('This is an unlabeled pkl file carrying pose information. The source video is unknown.', UserWarning)
            self.load_pkl(pose_path)
        else:
            poseobj = joblib.load(pose_path)
            self.vid_path = poseobj['vidpath']
            self.vid_name = poseobj['vid_name']
            self.framecount = poseobj['framecount']
            self.fps = poseobj['fps']
            self.video_time = poseobj['video_time']
            self.video_shape = poseobj['video_shape']
            if check_keys(poseobj, 'face_data'):
                self.face_data = pose_obj['face_data']
            if check_keys(poseobj, 'pose_data'):
                self.pose_data = poseobj['pose_data']






