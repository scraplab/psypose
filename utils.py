#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:51:18 2021

@author: f004swn
"""

import numpy as np
import ast
import time
import cv2
import pandas as pd
import face_recognition

def from_np_array(array_string):
    # converter function for interpreting face data csv
    array_string = ','.join(array_string.replace('[ ', '[').split())
    return np.array(ast.literal_eval(array_string))
    
def string2list(string):
    # converter function for interpreting face data csv
    if '.' in string:
        vals = [float(i) for i in string[1:-1].replace('.', '').split(' ') if i!='']
    else:
        vals = [float(i) for i in string[1:-1].replace(',', '').split(' ') if i!='']
    return vals
    

def ts_to_frame(ts, framerate):
    h, m, s = ts.split(':')
    conv_ts = (int(h)*60*60 + int(m)*60 + int(s))*framerate
    return round(conv_ts)
    
    #and possibly need to convert frame numbers to timestamps
def frame_to_ts(frame, fps):
    seconds = round(frame//fps)
    ts = time.strftime('%H:%M:%S', time.gmtime(seconds))
    return ts

def check_match(bod, fac):
    cx, cy, w, h = [float(i) for i in bod]
    top_b, right_b, bottom_b, left_b = [(cy-h/2), (cx+w/2), (cy+h/2), (cx-w/2)]
    top_f, right_f, bottom_f, left_f = [float(i) for i in fac]
    face_x = (right_f-left_f)/2 + left_f
    face_y = (bottom_f-top_f)/2 + top_f

    if (left_b < face_x < right_b and
        top_b < face_y < bottom_b):
        return True
    else:
        return False
    
def frame2array(frame_no, video_opened):
    video_opened.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
    ret, frame = video_opened.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def resize_image(array, newsize):
    array = cv2.resize(array, dsize=newsize, interpolation=cv2.INTER_CUBIC)
    array = np.expand_dims(array, axis=0)[0]
    return array

def crop_image(array, bbox):
    top, right, bottom, left = bbox
    new_img = array[top:bottom, left:right, :]
    return new_img

def evaluate_pred_ID(charList, ground, pred):
    
    #ground and pred need to be same-shape np arrays
    
    # might be better to take dataframes so columns
    # can be cross-referenced (less preprocessing of presence matrices)
    from sklearn.metrics import confusion_matrix
    chars = list(charList)
    chars.insert(0, 'metric')
    metrics = ['overall', 'true_positive', 'true_negative', 
               'false_positive', 'false_negative', 'true_presence_proportion']
    acc_df = pd.DataFrame(columns=chars)
    acc_df['metric'] = metrics

    for i, char in enumerate(chars[1:]):
        tn, fp, fn, tp = confusion_matrix(ground[:,i], pred[:,i]).ravel()
        overall = (tp+tn)/len(ground[:,i])
        t_pos_acc = tp/np.sum(ground[:,i])
        t_neg_acc = tn/(len(ground[:,i])-np.sum(ground[:,i]))
        # out of all predicted positives which are false
        false_pos = fp/np.sum(pred[:,i])
        false_neg = fn/(len(pred[:,i]) - np.sum(pred[:,i]))
        tot_pos = np.sum(ground[:,i]) / ground.shape[0]
        acc_df[char] = [overall, t_pos_acc, t_neg_acc, false_pos, false_neg, tot_pos]

    return acc_df

            
def get_data(trans, parameter):
    # Extracts desired data from pliers dataframe format
    enc = trans[parameter]
    arr = np.zeros((enc.shape[0], len(enc[0])))
    for frame in range(arr.shape[0]):
        arr[frame,:]=enc[frame]
    return arr

def default_encoding(face_array):
    # This is a janky implmementation. Right?
    resized_array = resize_image(face_array, (150, 150))
    encoding = face_recognition.face_encodings(resized_array)[0]
    return encoding

    