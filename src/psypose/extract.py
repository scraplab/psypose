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
import os.path as osp
import joblib
import numpy as np
import cv2
import glob
import pandas as pd
from tqdm import tqdm
import atexit
import shutil

from psypose.pose_estimation import estimate_pose

from psypose import utils
#from psypose.face_identification import add_face_id, cluster_ID

import sys

#from feat.detector import Detector

sys.path.append(os.getcwd())

def annotate(pose, face_box_model='mtcnn', au_model='rf', face_id_model='deepface', 
             every=1, output_path=None, save_results=True, shot_detection=False,
             person_tracking=False, extract_aus=True, extract_face_id=False, run_clustering=False, num_workers=None,
             smooth=True, image_folder=None):

    """
    Annotate a video with pose, facial expression, and face identity features.
    @param pose: PsyPose pose() object
    @param face_box_model: Which face detection model to use. See feat.detector.Detector for options.
    @param au_model: Which action unit detection model to use. See feat.detector.Detector for options.
    @param face_id_model: Which model to use to generate face embeddings.
    @param every: Number of frames to skip for face annotation. Default is 1 (every frame).
    @param output_path: Output directory.
    @param save_results: (Bool) Save results to output_path.
    @param shot_detection: (Bool) Whether to run shot detection.
    @param person_tracking: Whether to run person tracking using face identification (better for feature films).
    @param extract_aus: (Bool) Whether to extract facial expressions.
    @param extract_face_id: (Bool) Whether to extract face identity features.
    @param run_clustering: (Bool) Whether to run clustering on face identity features.
    @param num_workers: Number of workers to use for face detection. Default is None (uses all available).
    @param smooth: (Bool) Whether to apply smoothing via One Euro Filter to pose data.
    @param image_folder: Where to save temporary image files. Default is None (uses default location).
    @return: Pose object populated with annotations.
    """

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

    if not image_folder:
        image_folder = osp.join(utils.PSYPOSE_DATA_DIR, pose.vid_name)
    else:
        image_folder = osp.join(image_folder, pose.vid_name)
    pose.image_folder = image_folder

    ########## Run shot detection ##########
    pose.shot_detection = shot_detection
    if shot_detection:
        tqdm.write("Detecting shots...")
        shots = utils.get_shots(pose.vid_path)
        # Here, shots is a list of tuples (each tuple contains the in and out frames of each shot)
        #pose_data, pose.splitcount = utils.split_tracks(pose_data, shots)
        pose.shots = shots

     ########## Run pose estimation ##########
    pose.smooth = smooth
    pose_data = estimate_pose(pose)
    print("Processing output data...")
    # Split tracks based on shot detection

    #pose_data = smooth_pose_data(pose_data)  # applying one euro filter

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
     
     ########## Extract face identity encodings ##########

    # if extract_aus and extract_face_id:
    #     add_face_id(pose)

    ############ Run face encoding clustering for identification ####################


     
     ########## Saving results ##########
    if save_results:
        if extract_aus:
            pose.face_data.to_csv(os.path.join(output_path, 'psypose_faces.csv'))

    print('Finished annotation for file: ', os.path.basename(pose.vid_path))
    return pose
     
     


     

 
 