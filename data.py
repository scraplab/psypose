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

from pliers.stimuli import VideoStim
from pliers.filters import FrameSamplingFilter
from pliers.converters import VideoFrameCollectionIterator

import torch


from psypose.models import facenet_keras, deepface
class pose(object):
    
    def __init__(self):
        self.is_clustered = False
        self.clusters_named = False
        self.shots = None
        pass
            
    def load_fmri(self, fmri_path, TR):
        self.fmri_path = fmri_path
        self.brain_data = nib.load(fmri_path).get_fdata()
        self.TR = TR
        # all timeseries will be in milliseconds for accuracy.
        self.brain_time = [1000*TR for tr in range(self.brain_data.shape[0])]
        
    def load_face_data(self, face_data):
        self.face_data_path = os.path.abspath(face_data)
        
        face_data_array = np.genfromtxt(face_data, delimiter=',')
        n_frames = face_data_array.shape[0]
        face_df = pd.DataFrame(columns=['frame_ids', 'locations', 'encodings'])
        face_df['frame_ids'] = face_data_array[:,0]
        face_df['locations'] = [face_data_array[i,1:5] for i in range(n_frames)]
        face_df['encodings'] = [face_data_array[i,5:] for i in range(n_frames)]
        self.face_data = face_df
        
    def load_video(self, vid_path):
        vid_path = os.path.abspath(vid_path)
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

    def encode_faces(self, overwrite=False, encoder='default', use_TR=False, every=None, out=None):
        if every==None:
            # encode every frame if sampling parameter not defined
            every=1
            
        def get_data(trans, parameter):
            enc = trans[parameter]
            arr = np.zeros((enc.shape[0], len(enc[0])))
            for frame in range(arr.shape[0]):
                arr[frame,:]=enc[frame]
            return arr
        print('\nParsing video data...')
        vid = VideoStim(self.vid_path)
        vid_filt = FrameSamplingFilter(every=every).transform(vid)
        frame_conv = VideoFrameCollectionIterator()
        vid_frames = frame_conv.transform(vid_filt)
        
        if torch.cuda.is_available():
            ext_loc = FaceRecognitionFaceLocationsExtractor(model='cnn')
        else:
            ext_loc = FaceRecognitionFaceLocationsExtractor()
        #for this, change to model='cnn' for GPU use on discovery

        print("\nExtracting face locations...")
        # This is using pliers, which uses dlib for extraction face locations (SOTA)
        face_locs_data = ext_loc.transform(vid_frames)
        #remove frames without faces
        face_locs_data = [obj for obj in face_locs_data if len(obj.raw) != 0]
        # convert to dataframe and remove frames with no faces

        frame_ids = [obj.stim.frame_num for obj in face_locs_data]
        locations = [obj.raw for obj in face_locs_data if len(obj.raw) != 0]
        
        # expanding multi-face frames
        frame_ids_expanded = []
        bboxes_expanded = []
        face_imgs = []
        for f, frame in enumerate(frame_ids):
            face_bboxes = locations[f]
            frame_img = utils.frame2array(frame, self.video_cv2)
            for i in face_bboxes:
                frame_ids_expanded.append(frame)
                bboxes_expanded.append(i)
                face_imgs.append(utils.crop_image(frame_img, i))
                
        bboxes = np.array(bboxes_expanded)

        print("\nExtracting face encodings...")
        
        if encoder=='default':
            encoding_length = 128
            encode = utils.default_encoding
        elif encoder=='facenet':
            encoding_length = 128
            encode = facenet_keras.encode
        elif encoder=='deepface':
            encoding_length = 2622
            encode = deepface.encode
        
        out_array = np.empty((len(frame_ids_expanded), 5+encoding_length))
        encodings = []
        for image in tqdm(face_imgs):
            encodings.append(encode(image))
        out_array[:,0], out_array[:,1:5], out_array[:, 5:] = frame_ids_expanded, bboxes, np.array(encodings)
        
        face_df = pd.DataFrame(columns=['frame_ids', 'bboxes', 'encodings'])
        face_df['frame_ids'] = frame_ids_expanded
        face_df['bboxes'] = bboxes_expanded
        face_df['encodings'] = encodings
        
        self.face_data = face_df
        
        if out != None:
            print('Saving results...')
            np.savetxt(out, out_array, delimiter=',')
        self.faces_encoded = True
    
    ## add face locations and face encodings as separate self properties?

        
    # def consolidate_clusters(self):
    #     tot_frames = int(self.video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
