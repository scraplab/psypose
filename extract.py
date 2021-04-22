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
from pliers.extractors import merge_results, FaceRecognitionFaceLocationsExtractor, FaceRecognitionFaceEncodingsExtractor  
from pliers.stimuli import VideoStim
from pliers.filters import FrameSamplingFilter
from pliers.converters import VideoFrameCollectionIterator
import sys

sys.path.append(os.getcwd())

from psypose.models import facenet_keras, deepface

def annotate(pose, output_path=None, tracking_method='bbox', 
    vibe_batch_size=225, mesh_out=False, run_smplify=False, render=False, wireframe=False,
    sideview=False, display=False, save_obj=False, gpu_id=0, output_folder='MEVA_outputs',
    detector='yolo', yolo_img_size=416, exp='', cfg=''):
        
     # ========= Run shot detection ========= #
     
     shots = utils.get_shots(pose.vid_path)
     # Here, shots is a list of tuples (each tuple contains the in and out frames of each shot)
     
     # Run pose estimation and get pose data
     
     pose_data = estimate_pose(pose.vid_path)
     
     pose_data = utils.split_tracks(pose_data, shots)
     
     return pose_data


     

 
 