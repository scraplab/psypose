# this script is for extracting poses from my data!
import os
import numpy as np
from psypose import data, pose_estimation
from psypose import utils as putils
from pathlib import Path
import logging
import moviepy.editor as mp
import joblib
import time
import glob

data_dir = Path('/safestore/users/landry/SCRAP/data/andromeda_storage/conversations_unconstrained')

pose_names = ['cam1_pose.pkl', 'cam2_pose.pkl']

# get all the sessions in the data directory
def screen_paths(dir):
    sessions = [f for f in dir.iterdir() if f.is_dir()]
    sessions = [i for i in sessions if '_' in i.name]
    sessions = [i for i in sessions if '_remove' not in str(i)]
    # make sure both cam1 and cam2 are in derivatives
    sessions = [i for i in sessions if (i / 'derivatives' / 'cam1_concatenated_trimmed.mp4').exists()]
    sessions = [i for i in sessions if (i / 'derivatives' / 'cam2_concatenated_trimmed.mp4').exists()]
    
    for session in sessions:
        if not (session / 'processed').exists():
            os.makedirs(session / 'processed', exist_ok=True)

    # check if the session has the pose files. If it has two pose files, remove it from the list
    to_process = []
    for session in sessions:
        processed = str(session / 'processed')
        poses = glob.glob(processed + '/**_pose.pkl')
        poses = [Path(p) for p in poses]
        poses = [p for p in poses if p.exists()]
        if len(poses) != 2:
            to_process.append(session)

    return to_process

def downsample(vidpath, height=540):
    clip = mp.VideoFileClip(vidpath)
    clip_resized = clip.resize(height=height) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
    vidpath = Path(vidpath)
    outpath = str(vidpath.with_name(vidpath.stem + f'_ds{height}.mp4'))
    clip_resized.write_videofile(outpath)
    return outpath

def extract_pose(vpath, render=False, smooth=False):
    if not isinstance(vpath, Path):
        vpath = Path(vpath)
    opath = vpath.with_name(vpath.stem + '_pose.pkl')
    pose = data.Camera()
    pose.output_path = opath
    pose.load_video(vpath)
    pose.no_render = True
    pose.smooth = smooth
    estimate = pose_estimation.estimate_pose(pose)
    return estimate

sessions = screen_paths(data_dir)
print('##############################\n')
print(f'Found {len(sessions)} sessions to process:\n')
for i in sessions:
    print(str(i))

ready = input('Ready to process? (y/n): ')

if ready != 'y':
    exit()
else:
    print('Starting...\n')
    t0 = time.time()
    for s, session in enumerate(sessions):
        print(f'Processing {session.name} ({s+1}/{len(sessions)})')
        outpath = session / 'processed'
        # get the paths to each camera
        cam1 = session / 'derivatives' / 'cam1_concatenated_trimmed.mp4'
        cam2 = session / 'derivatives' / 'cam2_concatenated_trimmed.mp4'
        # check if they have downsampled versions
        cam1_ds = session / 'derivatives' / 'cam1_concatenated_trimmed_ds540.mp4'
        cam2_ds = session / 'derivatives' / 'cam2_concatenated_trimmed_ds540.mp4'
        if cam1_ds.exists() and cam2_ds.exists():
            print('Downsampled videos found, skipping downsample step')
            cam1 = cam1_ds
            cam2 = cam2_ds
        else:
        # downsample the videos, return the new paths to the downsampled videos
        # dont downsample if the videos are already downsampled
            print('Downsampling videos...')
            cam1 = downsample(str(cam1))
            cam2 = downsample(str(cam2))
        # extract the poses from the downsampled videos
        try:
            p1, p2 = extract_pose(cam1), extract_pose(cam2)
        except Exception as e:
            print(f'Error processing {session.name}: {e}')
        # saving the poses in the processed directory
        joblib.dump(p1, outpath / 'cam1_pose.pkl')
        joblib.dump(p2, outpath / 'cam2_pose.pkl')
        print(f'Processed {session.name} in {time.time() - t0} seconds')

print('Done!')
