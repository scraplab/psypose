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
#import face_recognition
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from tqdm import tqdm
import os.path as osp
import os
import subprocess
import traceback
from pathlib import Path
import gdown
from requests.exceptions import MissingSchema

def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/scratch', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder

def from_np_array(array_string):
    # this is old
    if 'e' in array_string:
        out = array_string.strip('[]').split(' ')
        out = [i.strip('\n') for i in out]
        out = [ast.literal_eval(i) for i in out]
        return out
    # converter function for interpreting face data csv
    else:
        array_string = ','.join(array_string.replace('[ ', '[').split())
    return array_string
#return np.array(ast.literal_eval(array_string))

    
def string2list(string):
    # converter function for interpreting face data csv
    if '.' in string:
        vals = [float(i) for i in string[1:-1].replace('.', '').split(' ') if i!='']
    else:
        vals = [float(i) for i in string[1:-1].replace(',', '').split(' ') if i!='']
    return vals

def string_is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def ts_to_frame(ts, framerate):
    # converts a timestamp to frame
    h, m, s = ts.split(':')
    conv_ts = (int(h)*60*60 + int(m)*60 + int(s))*framerate
    return round(conv_ts)
    
def frame_to_ts(frame, fps):
    seconds = round(frame//fps)
    ts = time.strftime('%H:%M:%S', time.gmtime(seconds))
    return ts

def check_match(bod, fac):
    # need to update this because of newly added py-feat face detection
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
    # returns a video from as a numpy array in uint8
    video_opened.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
    ret, frame = video_opened.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.destroyAllWindows()
    return frame

def resize_image(array, newsize):
    array = cv2.resize(array, dsize=newsize, interpolation=cv2.INTER_CUBIC)
    array = np.expand_dims(array, axis=0)[0]
    return array

def crop_image(array, bbox):
    top, right, bottom, left = bbox
    new_img = array[top:bottom, left:right, :]
    return new_img

def crop_image_wh(array, data):
    # The _wh means it is taking the py-feat face bounding box format
    # Top-left corner x and y, then the width and height of the box from that point
    cx, cy, w, h = [i for i in data]
    # I don't think this calculation is right:
    top, right, bottom, left = [int(round(i)) for i in [(cy-h/2), int(cx+w/2), int(cy+h/2), (cx-w/2)]]
    new_img = array[top:bottom, left:right, :]
    return new_img

def crop_image_body(array, data):
    # you can now just use on crop image function because the body and face bboxes are in the same format
    cx, cy, w, h = [i for i in data]
    top, right, bottom, left = [int(round(i)) for i in [(cy-h/2), int(cx+w/2), int(cy+h/2), (cx-w/2)]]
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
        # percent of positives that are true
        t_pos_acc = tp/np.sum(ground[:,i])
        # percent of negatives that are true
        t_neg_acc = tn/(len(ground[:,i])-np.sum(ground[:,i]))
        # perent of positives that are false
        false_pos = fp/np.sum(pred[:,i])
        # percent of negatives that are false
        false_neg = fn/(len(pred[:,i]) - np.sum(pred[:,i]))
        # True proportion of all frames that are positive
        tot_pos = np.sum(ground[:,i]) / ground.shape[0]
        acc_df[char] = [overall, t_pos_acc, t_neg_acc, false_pos, false_neg, tot_pos]

    return acc_df

            


#def default_encoding(face_array):
#    # This is a janky implmementation. Right?
#    resized_array = resize_image(face_array, (150, 150))
#    encoding = face_recognition.face_encodings(resized_array)[0]
#    return encoding

def write2vid(img_arr, fps, out_name, out_size):
    """
    Takes numpy array and writes each frame to a video file.
    """
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)
    for i in range(len(img_arr)):
        print(str(i+1)+'/'+str(len(img_arr)))
        img_color = cv2.cvtColor(img_arr[i], cv2.COLOR_BGR2RGB)
        out.write(img_color)
    cv2.destroyAllWindows()
    out.release()
    
def get_shots(video_path, downscale_factor=None, threshold=30):
    """
    

    Parameters
    ----------
    video_path : scenedetect VideoManager object.
    downscale_factor : Factor by which to downscale video to improve speed.
    threshold : Cut detection threshold.

    Returns
    -------
    cut_tuples : A list of tuples where each tuple contains the in- and out-frame of each shot.

    """
    print('\nDetecting cuts...')
    video = VideoManager([video_path])
    video.set_downscale_factor(downscale_factor)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video.start()
    scene_manager.detect_scenes(frame_source=video)
    shot_list = scene_manager.get_scene_list()
    cut_tuples = []
    for shot in shot_list:
        cut_tuples.append((shot[0].get_frames(), shot[1].get_frames()))
    video.release()
    return cut_tuples

def video_to_array(cap):
    
    """
    Takes video file and converts it into a numpy array with uint8 encoding.
    
    cap : either a numpy array or scenedetect.VideoManager object
    """
    if isinstance(cap, VideoManager):
        cap.start()
    
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, h, w, 3), np.dtype('uint8'))
    # Initialize frame counter
    fc = 0
    ret = True
    pbar = tqdm(total=frameCount)
    print("\nLoading video into memory...\n")
    while (fc < frameCount  and ret):
        # cap.read() returns a bool (ret) if the frame was retrieved along with the frame as a numpy array
        ret, frame = cap.read()
        # cv2 reads images as blue-green-red, which needs to be converted into red-green-blue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # fill empty array with video frame
        buf[fc] = frame
        fc += 1
        pbar.update(1)
    pbar.close()

    cap.release()

    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    return buf

def find_nearest(array, value):
    # finds the nearest value in an array and returns the index
    # used for finding a frame from approximate timestamp
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def split_track(idx, track):
    # splits all items in a pose estimation track based on a detected cut
    A, B = {}, {}
    for key in list(track.keys()):
        A[key], B[key] = track[key][:idx], track[key][idx:]
    return A, B

def split_tracks(data, shots):
    tracks_split = []
    for track in data:
        split = False
        frames = data[track]['frame_ids']
        for shot in shots:
            out = shot[1]-1
            if (out in frames) and (frames[-1] != out): # if the last frame in a shot is not the last frame in a track,
                split = True
                cut_idx = np.where(frames==out)[0][0] + 1 # not sure if this is the problem
                first_half, second_half = split_track(cut_idx, data[track])
                split_track = [first_half, second_half]
        if split:
            for sp in split_track:
                tracks_split.append(sp)
        else:
            tracks_split.append(data[track])

    out = {}
    for i, track in enumerate(tracks_split):
        out[i] = track
    return out

def get_bbox(row):
    # gets bboxes from py-feat output
    x, y, w, h = row[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']]
    return [x, y, w, h]
    
def crop_face(array, data):
    cx, cy, w, h = [i for i in data]
    top, right, bottom, left = [int(round(i)) for i in [cy, (cx+w), (cy+h), cx]]
    new_img = array[top:bottom, left:right, :]
    return new_img

# Paxton Fitzpatrick authored this section
PSYPOSE_DATA_FILES = {
    'facenet_keras.h5': '1eyE-IIHpkswHhYnPXX3HByrZrSiXk00g',
    'vgg_face_weights.h5': '1AkYZmHJ_LsyQYsML6k72A662-AdKwxsv',
    'meva_data.zip': '1l5pUrV5ReapGd9uaBXrGsJ9eMQcOEqmD'
    }

# Here, the ~ is referencing the user's HOME directory, this is syntax for expanduser(),
# which references the user's file system.

PSYPOSE_DATA_DIR = Path('~/.psypose').expanduser()

def check_data_files(prompt_confirmation=False):
    missing_files = PSYPOSE_DATA_FILES.copy()
    if PSYPOSE_DATA_DIR.is_dir():
        for fname in PSYPOSE_DATA_FILES.keys():
            expected_loc = PSYPOSE_DATA_DIR.joinpath(fname)
            if expected_loc.suffix in {'zip', 'gz', 'tgz', 'bz2'}:
                expected_loc = expected_loc.with_suffix('')
            if expected_loc.exists():
                missing_files.pop(fname)
    if any(missing_files):
        if prompt_confirmation:
            msg = (
                      f"Psypose needs to download {len(missing_files)} files "
                      f"in order to run:\n\t{', '.join(missing_files.keys())}"
                      "\n\tDo you want to download them now?\n[Y/n] \n"
            )
            while True:
                response = input(msg).lower().strip()
                if response in ('y', ''):
                    confirmed = True
                    break
                elif response == 'n':
                    confirmed = False
                    break
        else:
            confirmed = True
        if confirmed:
            print('\nDownloading Psypose model weights...')
            if not PSYPOSE_DATA_DIR.is_dir():
                print(f"creating {PSYPOSE_DATA_DIR} ...")
                PSYPOSE_DATA_DIR.mkdir(parents=False, exist_ok=False)
            errors = {}
            for fname, gdrive_id in missing_files.items():
                dest_path = PSYPOSE_DATA_DIR.joinpath()
                print(f"downloading {fname} ...")
                try:
                    download_from_gdrive(gdrive_id, dest_path)
                except (MissingSchema, OSError) as e:
                    errors[item[0]] = e

            if any(errors):
                print(
                         f"Failed to download {len(errors)} files. See stack "
                         f"trace{'s' if len(errors) > 1 else ''} below for "
                         "more info:\n"
                )
                for fname, e in errors.items():
                    print(f"{fname.upper()}:")
                    traceback.print_exception(type(e), e, e.__traceback__)
                    print('='*40, end='\n\n')
        else:
            warnings.warn(
                             "missing required files. Some Psypose "
                             "functionality may be unavailable"
            )

def download_from_gdrive(gdrive_id, dest_path):
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    gdown.download(url, str(dest_path), quiet=False)
    if dest_path.suffix in {'zip', 'gz', 'tgz', 'bz2'}:
        print(f"extracting {dest_path} ...")
        gdown.extractall(str(dest_path))
        print(f"removing {dest_path} ...")
        dest_path.unlink()


    
    