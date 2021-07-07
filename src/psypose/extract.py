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
from tqdm import tqdm

from psypose import utils
#from psypose.pose_estimation import estimate_pose
from psypose.ROMP.video_romper import estimate_pose
from psypose.face_identification import add_face_id

import sys

from feat.detector import Detector

sys.path.append(os.getcwd())


def annotate(pose, face_box_model='mtcnn', au_model='rf', face_id_model='deepface', 
             every=1, output_path=None, save_results=True, shot_detection=True, extract_aus=True, extract_face_id=True, num_workers=None):

     
     ########## Run pose estimation ##########
     
     pose_data = estimate_pose(pose, num_workers=num_workers) 
     # Split tracks based on shot detection


     ########## Run shot detection ##########
     
     if shot_detection:
        tqdm.write("Detecting shots...")
        shots = utils.get_shots(pose.vid_path)
     # Here, shots is a list of tuples (each tuple contains the in and out frames of each shot)
        pose_data, pose.splitcount = utils.split_tracks(pose_data, shots)
        pose.shots = shots

          
     # Add pose data to the pose object
     pose.pose_data = pose_data
     pose.n_tracks = len(pose_data)
     
     
     ########## Run face detection + face feature extraction ##########

     if extract_aus:
        detector = Detector(face_model = face_box_model, au_model = au_model)
        tqdm.write("Extracting facial expressions...")
        pose.face_data = detector.detect_video(pose.vid_path, skip_frames = every)
     
     ########## Extract face identify encodings ##########

     if extract_aus and extract_face_id:
        add_face_id(pose)
     
     ########## Saving results ##########
     if output_path==None:
         output_path = os.getcwd()
         
     if save_results:
         os.makedirs(output_path+'/'+pose.vid_name, exist_ok=True)
         if extract_aus:
            pose.face_data.to_csv(output_path+'/'+pose.vid_name+'/psypose_faces.csv')
         joblib.dump(pose.pose_data, os.path.join(output_path+'/'+pose.vid_name+'/psypose_bodies.pkl'))

     print('Finished annotation for file: ', pose.vid_name)
     
     


     

 
 