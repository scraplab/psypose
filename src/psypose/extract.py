#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:19:37 2021

@author: f004swn
"""

"""
Tools for extracting pose features from a pose object.
"""

import os
import joblib
import numpy as np
import cv2
import glob
import pandas as pd

from psypose import utils
from psypose.pose_estimation import estimate_pose
from psypose.face_identification import add_face_id

import sys

from feat.detector import Detector

sys.path.append(os.getcwd())


def annotate(pose, face_box_model='mtcnn', au_model='rf', face_id_model='deepface', 
             every=1, output_path=None, save_results=True):
        
     ########## Run shot detection ##########
     
     print("\nDetecting shots...")
     shots = utils.get_shots(pose.vid_path)
     # Here, shots is a list of tuples (each tuple contains the in and out frames of each shot)
     
     ########## Run pose estimation ##########
     
     pose_data = estimate_pose(pose, num_workers=2) # This is best for colab notebooks
     # Split tracks based on shot detection
     pose_data, pose.splitcount = utils.split_tracks(pose_data, shots)
          
     # Add pose data to the pose object
     pose.pose_data = pose_data
     pose.n_tracks = len(pose_data)
     pose.shots = shots
     
     ########## Run face detection + face feature extraction ##########
     
     detector = Detector(face_model = face_box_model, au_model = au_model)
     
     pose.face_data = detector.detect_video(pose.vid_path, skip_frames = every)
     
     ########## Extract face identify encodings ##########
     
     add_face_id(pose)
     
     ########## Saving results ##########
     if output_path==None:
         output_path = '../'
         
     if save_results:
         os.makedirs(output_path+'/'+pose.vid_name, exist_ok=True)
         pose.face_data.to_csv(output_path+'/'+pose.vid_name+'/psypose_faces.csv')
         joblib.dump(pose.pose_data, os.path.join(output_path+'/'+pose.vid_name+'/psypose_bodies.pkl'))
     
     


     

 
 