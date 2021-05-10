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
from psypose.face_identification 

from pliers.extractors import merge_results, FaceRecognitionFaceLocationsExtractor, FaceRecognitionFaceEncodingsExtractor  
from pliers.stimuli import VideoStim
from pliers.filters import FrameSamplingFilter
from pliers.converters import VideoFrameCollectionIterator
import sys

from py-feat.detector import Detector

sys.path.append(os.getcwd())


def annotate(pose, face_encoding_model='deepface', every=1, output_path=None):
        
     ########## Run shot detection ##########
     
     shots = utils.get_shots(pose.vid_path)
     # Here, shots is a list of tuples (each tuple contains the in and out frames of each shot)
     
     ########## Run pose estimation ##########
     
     pose_data = estimate_pose(pose)
     # Split tracks based on shot detection
     pose_data = utils.split_tracks(pose_data, shots)
     
     pose.pose_data = pose_data
     
     # Add pose data to the pose object
     pose.pose_data = pose_data
     pose.n_tracks = len(pose_data)
     pose.shots = shots
     
     ########## Run face detection + face feature extraction ##########
     
     detector = Detector(face_model = "retinaface", au_model = "rf")
     
     pose.face_data = detector.detect_video(pose.vid_path, every=every)
     
     ########## Extract face identify encodings ##########
     
     add_face_id(pose)
     
     ########## Saving results ##########
     if output_path==None:
         output_path = pose.vid_name
         os.makedirs(output_path, exist_ok=True)
         
     pose.face_data.to_csv('psypose_faces.csv')
     joblib.dump(pose.pose_data, os.path.join(output_path, 'psypose_bodies.pkl'))
     
    
     return pose_data


     

 
 