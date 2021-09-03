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

from ROMP_psypose.core.test import estimate_pose

from psypose import utils
from psypose.augment import gather_tracks, smooth_pose_data, add_quaternion
from psypose.face_identification import add_face_id

import sys

from feat.detector import Detector

sys.path.append(os.getcwd())


def annotate(pose, face_box_model='mtcnn', au_model='rf', face_id_model='deepface', 
             every=1, output_path=None, save_results=True, shot_detection=True, extract_aus=True, extract_face_id=True, num_workers=None):

    # if output path is not defined, a directory named after the input video will be created in whatever directory the script is ran.
     if output_path==None:
         this_dir = os.getcwd()
         output_path = os.path.join(this_dir, pose.vid_name)
         os.makedirs(output_path, exist_ok=True)
         pose.output_path = output_path
     else:
         pose.output_path = output_path

    if save_results:
        os.makedirs(output_path, exist_ok=True)

     ########## Run pose estimation ##########
     
     pose_data = estimate_pose(pose)
     print("Stitching tracks...")
     pose_data = gather_tracks(pose_data)
     pose_data = add_quaternion(pose_data)
     pose_data = smooth_pose_data(pose_data) #applying one euro filter
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

     if save_results:
        joblib.dump(pose.pose_data, os.path.join(output_path, 'psypose_bodies.pkl'))

     ########## Run face detection + face feature extraction ##########

     if extract_aus:
        detector = Detector(face_model = face_box_model, au_model = au_model)
        tqdm.write("Extracting facial expressions...")
        pose.face_data = detector.detect_video(pose.vid_path, skip_frames = every)
     
     ########## Extract face identify encodings ##########

     if extract_aus and extract_face_id:
        add_face_id(pose)
     
     ########## Saving results ##########
     if save_results:
         if extract_aus:
            pose.face_data.to_csv(os.path.join(output_path, 'psypose_faces.csv'))

     print('Finished annotation for file: ', os.path.basename(pose.vid_path))
     
     


     

 
 