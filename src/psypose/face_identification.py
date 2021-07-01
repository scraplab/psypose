#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:30:19 2021

@author: f004swn
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from psypose import utils
from psypose.models import facenet_keras, deepface

def add_face_id(pose, overwrite=False, encoder='facenet', use_TR=False, out=None):
    
    face_df = pd.DataFrame(pose.face_data)
    # Remove possible nan rows
    face_df = face_df.dropna(axis=0)
    #removing negative values
    face_df = face_df[face_df['FaceRectY']>=0]
    face_df = face_df[face_df['FaceRectX']>=0]
    faces_to_process = int(face_df.shape[0])
    unique_frames = [int(i) for i in np.unique(face_df['frame'])]

    #if encoder=='default':
    #    encoding_length = 128
    #    encode = utils.default_encoding
    if encoder=='facenet':
        encoding_length = 128
        encode = facenet_keras.encode
    elif encoder=='deepface':
        encoding_length = 2622
        encode = deepface.encode
    
    encoding_array = np.empty((faces_to_process, encoding_length))
        
    print("Encoding face identities...\n", flush=True)
    pbar = tqdm(total=faces_to_process)
    counter = -1
    for frame in unique_frames:
        img = utils.frame2array(frame, pose.video_cv2)
        sub = face_df[face_df['frame']==frame]
        for loc in range(sub.shape[0]):
            row = sub.iloc[loc]
            bbox = row[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']]
            face_cropped = utils.crop_face(img, bbox)
            encoding = encode(face_cropped)
            counter+=1
            encoding_array[counter] = encoding
            pbar.update(1)
    pbar.close()
            
    encoding_columns = ['enc'+str(i) for i in range(encoding_length)]
    face_df[encoding_columns] = encoding_array
    
    face_df = face_df.reset_index(drop=True)

    pose.face_data = face_df
    pose.faces_encoded = True
