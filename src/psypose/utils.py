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
import zipfile
from sklearn.metrics import confusion_matrix
from PIL import Image
import base64
from io import BytesIO
import shutil
import torchvision
import torch
from psypose.pare_dl.download_pare_models import install_pare_models, pare_status

def convert_cam_to_3d_trans(cams, weight=2.):
    trans3d = []
    (s, tx, ty) = cams
    depth, dx, dy = 1./s, tx/s, ty/s
    trans3d = np.array([dx, dy, depth])*weight
    return trans3d

def img_preprocess(image, imgpath, input_size=512, ds='internet', single_img_input=False):
    image = image[:, :, ::-1]
    image_size = image.shape[:2][::-1]
    image_org = Image.fromarray(image)

    resized_image_size = (float(input_size) / max(image_size) * np.array(image_size) // 2 * 2).astype(np.int)
    padding = tuple((input_size - resized_image_size) // 2)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([resized_image_size[1], resized_image_size[0]], interpolation=3),
        torchvision.transforms.Pad(padding, fill=0, padding_mode='constant'),
        # torchvision.transforms.ToTensor(),
    ])
    image = torch.from_numpy(np.array(transform(image_org))).float()

    padding_org = tuple((max(image_size) - np.array(image_size)) // 2)
    transform_org = torchvision.transforms.Compose([
        torchvision.transforms.Pad(padding_org, fill=0, padding_mode='constant'),
        torchvision.transforms.Resize((input_size * 2, input_size * 2), interpolation=3),
        # max(image_size)//2,max(image_size)//2
        # torchvision.transforms.ToTensor(),
    ])
    image_org = torch.from_numpy(np.array(transform_org(image_org)))
    padding_org = (np.array(list(padding_org)) * float(input_size * 2 / max(image_size))).astype(np.int)
    if padding_org[0] > 0:
        image_org[:, :padding_org[0]] = 255
        image_org[:, -padding_org[0]:] = 255
    if padding_org[1] > 0:
        image_org[:padding_org[1]] = 255
        image_org[-padding_org[1]:] = 255

    offsets = np.array([image_size[1], image_size[0], resized_image_size[0], \
                        resized_image_size[0] + padding[1], resized_image_size[1], resized_image_size[1] + padding[0],
                        padding[1], \
                        resized_image_size[0], padding[0], resized_image_size[1]], dtype=np.int)
    offsets = torch.from_numpy(offsets).float()

    name = os.path.basename(imgpath)

    if single_img_input:
        image = image.unsqueeze(0).contiguous()
        image_org = image_org.unsqueeze(0).contiguous()
        offsets = offsets.unsqueeze(0).contiguous()
        imgpath, name, ds = [imgpath], [name], [ds]
    input_data = {
        'image': image,
        'image_org': image_org,
        'imgpath': imgpath,
        'offsets': offsets,
        'name': name,
        'data_set': ds}
    return input_data


def save_meshes(reorganize_idx, outputs, output_dir, smpl_faces):
    vids_org = np.unique(reorganize_idx)
    for idx, vid in enumerate(vids_org):
        verts_vids = np.where(reorganize_idx==vid)[0]
        img_path = outputs['meta_data']['imgpath'][verts_vids[0]]
        obj_name = os.path.join(output_dir, '{}'.format(os.path.basename(img_path))).replace('.mp4','').replace('.jpg','').replace('.png','')+'.obj'
        for subject_idx, batch_idx in enumerate(verts_vids):
            save_obj(outputs['verts'][batch_idx].detach().cpu().numpy().astype(np.float16), \
                smpl_faces,obj_name.replace('.obj', '_{}.obj'.format(subject_idx)))

def get_video_bn(video_file_path):
    return os.path.basename(video_file_path)\
    .replace('.mp4', '').replace('.avi', '').replace('.webm', '').replace('.gif', '')

def img_to_b64(arr_img):
    pil_img = Image.fromarray(arr_img)
    prefix = "data:image/png;base64,"
    with BytesIO() as stream:
        pil_img.save(stream, format="png")
        base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
    return base64_string

#def video_to_bytes(vid_arr):
#    out_strings = []
#    for img in range(vid_arr.shape[0]):
#        out_strings.append(img_to_b64(vid_arr[img]))
#    return out_strings

def bytes_to_arr(bString):
    r = base64.b64decode(bString, + "==")
    q = np.frombuffer(r, dtype=np.float64)
    return q


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
    bcx, bcy, bw, bh = [float(i) for i in bod] # body corner is bottom left
    fcx, fcy, fw, fh = [float(i) for i in fac] # face corner is top left

    top_b, right_b, bottom_b, left_b = [(bcy-bh/2), (bcx+bw/2), (bcy+bh/2), (bcx-bw/2)]
    top_f, right_f, bottom_f, left_f = [fcy, fcx+fw, fcy+fh, fcx]

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
    #cv2.destroyAllWindows()
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
    #cx, cy, w, h = [float(i) for i in data]
    #top, right, bottom, left = [int(round(i)) for i in [(cy - h / 2), (cx + w / 2), (cy + h / 2), (cx - w / 2)]]
    cx, cy, w, h = [i for i in data]
    top, right, bottom, left = [int(round(i)) for i in [cy, cx+w, cy+h, cx]]
    new_img = array[top:bottom, left:right, :]
    return new_img

def evaluate_pred_ID(charList, ground, pred):
    
    #ground and pred need to be same-shape np arrays
    
    # might be better to take dataframes so columns
    # can be cross-referenced (less preprocessing of presence matrices)
    
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
    print("Rendering...\n")
    out = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, out_size)
    for i in tqdm(range(len(img_arr))):
        img_color = cv2.cvtColor(img_arr[i], cv2.COLOR_BGR2RGB)
        out.write(img_color)
    #cv2.destroyAllWindows()
    out.release()


    
def get_shots(video_path, downscale_factor=None, threshold=12.0):
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
    #print('Detecting cuts...')
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
    #if isinstance(cap, VideoManager):
    #    cap.start()
    
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, h, w, 3), np.dtype('uint8'))
    # Initialize frame counter
    fc = 0
    ret = True
    #tqdm.write("Loading video...")
    pbar = tqdm(total=frameCount)
    #print("\nLoading video into memory...\n")
    while (fc < frameCount  and ret):
        # cap.read() returns a bool (ret) if the frame was retrieved along with the frame as a numpy array
        ret, frame = cap.read()
        # cv2 reads images as blue-green-red, which needs to be converted into red-green-blue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # fill empty array with video frame
        buf[fc] = frame
        fc += 1
        pbar.update(1)
        del frame
    pbar.close()

    #cap.release()

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return buf

def video_to_bytes(cap):
    
    """
    Takes video file and converts it into a list of b64 encodings
    
    cap : either a numpy array or scenedetect.VideoManager object
    """
    if isinstance(cap, VideoManager):
        cap.start()
    
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Initialize frame counter
    fc = 0
    ret = True
    pbar = tqdm(total=frameCount)
    #print("Encoding video...\n", flush=True)
    out_bytes = []
    tqdm.write("Encoding video...")
    while (fc < frameCount  and ret):
        # cap.read() returns a bool (ret) if the frame was retrieved along with the frame as a numpy array
        ret, frame = cap.read()
        # cv2 reads images as blue-green-red, which needs to be converted into red-green-blue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # fill empty array with video frame
        out_bytes.append(img_to_b64(frame))
        del frame
        fc += 1
        pbar.update(1)
    pbar.close()

    cap.release()

    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    return out_bytes

def slice_video(cap, frames):
    frameCount = len(frames)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frameCount, h, w, 3), np.dtype('uint8'))

    for f, frame_no in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no)
        ret, frame = cap.read()
        if not ret:
          break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf[f] = frame

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
    split_frames = []
    num_splits = 0
    track_labels = list(data.keys())
    # frame key could be either 'frames' or 'frame_ids' - checking that here
    frame_label = str([i for i in list(data[track_labels[0]].keys()) if 'frame' in i][0])
    #frame_label='frames'
    for track in track_labels:
        split = False
        frames = data[track][frame_label]
        for shot in shots:
            out = shot[1]-1
            if (out in frames) and (frames[-1] != out): # if the last frame in a shot is not the last frame in a track,
                split = True
                cut_idx = np.where(frames==out)[0][0] + 1
                first_half, second_half = split_track(cut_idx, data[track])
                track_segmented = [first_half, second_half]
        if split:
            for sp in track_segmented:
                tracks_split.append(sp)
                split_frames.append(out)
                num_splits += 1
        else:
            tracks_split.append(data[track])

    out = {}
    for i, track in enumerate(tracks_split):
        out[i] = track
    return out, num_splits, split_frames

def get_bbox(row):
    # gets bboxes from py-feat output
    # bottom left corner
    cx, cy, w, h = row[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']]
    return [cx, cy, w, h]
    
def crop_face(array, data):
    # I think this is where my problem is. 
    cx, cy, w, h = [i for i in data]
    top, right, bottom, left = [int(round(i)) for i in [cy, (cx+w), (cy+h), cx]]
    new_img = array[top:bottom, left:right, :]
    return new_img

def get_framecount(video_path):
    cap = cv2.VideoCapture(video_path)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    del cap
    return framecount

def get_framecount(vid_path):
    cap = cv2.VideoCapture(vid_path)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    del cap
    return fc

def get_image_shape(vid_path):
    cap = cv2.VideoCapture(vid_path)
    shape = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    del cap
    return shape

def track_presence_per_frame(pose):

    """
    Determine which tracks are present in each frame.
    @param pose: PsyPose pose object.
    @type pose: Pose
    @return: Dictionary where keys are frame IDs and values are lists of track IDs.
    @rtype: dict
    """

    frames = [i for i in range(pose.framecount)]
    track_lists = []
    for frame in frames:
        tracks_present = []
        for track, data in pose.pose_data.items():
           if frame in data['frame_ids']:
               tracks_present.append(track)
        track_lists.append(tracks_present)

    track_presence = dict(zip(frames, track_lists))
    if not hassattr(pose, 'track_presence'):
        pose.track_presence = track_presence
    return track_presence




def slice_pose(pose, frame_range):

    """
    Split pose data based on frame range.
    @param pose: PsyPose pose object
    @type pose: Pose
    @param frame_range: In frame and out frame.
    @type frame_range: Two-length iterable.
    @return: Split pose data.
    @rtype: dict
    """

    track_presence = track_presence_per_frame(pose)
    frame_span = [i for i in range(frame_range[0], frame_range[1])]
    tracks_subset = [track_presence[i] for i in frame_span] # list of lists
    allFrames = []
    for trackList in tracks_subset:
        for track in trackList:
            allFrames.extend(list(pose.pose_data[track]['frame_ids']))


def make_presence_mat(pose):
    dat = pose.pose_data.copy()
    framecount = pose.framecount
    dk = list(dat.keys())  # list of tracks
    ntt = len(dk)  # number of tracks
    ptmat = np.zeros((framecount, ntt))  # ntracks by nframes

    # make presence matrix
    for i in range(ntt):
        ptmat[dat[dk[i]]['frame_ids'], i] = 1
    return ptmat


PSYPOSE_DATA_FILES = {
    'facenet_keras.h5': '1eyE-IIHpkswHhYnPXX3HByrZrSiXk00g',
    'vgg_face_weights.h5': '1AkYZmHJ_LsyQYsML6k72A662-AdKwxsv'
}

PSYPOSE_DATA_DIR = Path('~/.psypose').expanduser()

def check_data_files(prompt_confirmation=False):
    missing_files = PSYPOSE_DATA_FILES.copy()
    if PSYPOSE_DATA_DIR.is_dir():
        for fname in PSYPOSE_DATA_FILES.keys():
            expected_loc = PSYPOSE_DATA_DIR.joinpath(fname)
            if expected_loc.suffix in {'.zip', '.gz', '.tgz', '.bz2'}:
                expected_loc = expected_loc.with_suffix('')
            if expected_loc.exists():
                missing_files.pop(fname)
    if any(missing_files):
        if prompt_confirmation:
            msg = (
                f"Psypose needs to download {len(missing_files)} "
                f"file{'s' if len(missing_files) > 1 else ''} in order "
                f"to run:\n\t{', '.join(missing_files.keys())}"
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
            if not PSYPOSE_DATA_DIR.is_dir():
                print(f"creating {PSYPOSE_DATA_DIR} ...")
                PSYPOSE_DATA_DIR.mkdir(parents=False, exist_ok=False)
            errors = {}
            for fname, url in missing_files.items():
                # dest_path = PSYPOSE_DATA_DIR.joinpath(fname)
                print(f"downloading {fname} ...")
                try:
                    download_file_from_web(url, fname)
                except (MissingSchema, OSError) as e:
                    errors[fname[0]] = e
            if any(errors):
                print(
                    f"Failed to download {len(errors)} files. See stack "
                    f"trace{'s' if len(errors) > 1 else ''} below for "
                    "more info:\n"
                )
                for fname, e in errors.items():
                    print(f"{fname.upper()}:")
                    traceback.print_exception(type(e), e, e.__traceback__)
                    print('=' * 40, end='\n\n')
        else:
            warnings.warn(
                "missing required files. Some Psypose "
                "functionality may be unavailable"
            )

def check_pare_install():
    if not pare_status:
        msg = (
            f"Pare needs to download model weights in order to run. Do you want to download them now?\n[Y/n] \n "
        )
        response = input(msg).lower().strip()
        if response in ('y', ''):
            confirmed = True
        elif response == 'n':
            confirmed = False
        else:
            confirmed = True
        if confirmed:
            install_pare_models()
        else:
            print('Psypose will try to download PARE in the future.\n\n')
    else:
        None


def download_file_from_web(url, filename):
    dest_path = PSYPOSE_DATA_DIR.joinpath(filename)
    urllib.request.urlretrieve(url, dest_path)
    if dest_path.suffix in {'.zip', '.gz', '.tgz', '.bz2'}:
        print(f"extracting {dest_path} ...")
        z = zipfile.ZipFile(str(dest_path))
        z.extractall(PSYPOSE_DATA_DIR)
        # zipfile.extractall(str(dest_path))
        print(f"removing {dest_path} ...")
        dest_path.unlink()


