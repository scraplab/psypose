# this script is for extracting poses from my data!
import os
import numpy as np
from psypose import extract, data, pose_estimation
from psypose import utils as putils
from pathlib import Path
import logging
import moviepy.editor as mp
import joblib
import time
import concurrent.futures

#data_dir = Path('safestore/users/landry/SCRAP/data/andromeda_storage/conversations_unconstrained')
data_dir = Path('/media/landry/fastscratch/landry_dev_scratch')

# get all the sessions in the data directory
def screen_paths(dir):
    sessions = [f for f in dir.iterdir() if f.is_dir()]
    sessions = [i for i in sessions if '_' in i.name]
    sessions = [i for i in sessions if len(i.name.split('_')[1]) == 3]
    return sessions

def downsample(vidpath, height=540):
    clip = mp.VideoFileClip(vidpath)
    clip_resized = clip.resize(height=height) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
    vidpath = Path(vidpath)
    outpath = str(vidpath.with_name(vidpath.stem + f'_ds{height}.mp4'))
    clip_resized.write_videofile(outpath)
    return outpath

def extract_pose(vpath, render=False):
    if not isinstance(vpath, Path):
        vpath = Path(vpath)
    opath = vpath.with_name(vpath.stem + '_pose.pkl')
    pose = data.pose()
    pose.load_video(vpath)
    pose.no_render = not render
    estimate = pose_estimation.estimate_pose(pose)
    return estimate

sessions = screen_paths(data_dir)

def process_session(session):
    t0 = time.time()
    outpath = session / 'processed'
    # get the paths to each camera
    cam1 = session / 'derivatives' / 'cam1_concatenated_trimmed.mp4'
    cam2 = session / 'derivatives' / 'cam2_concatenated_trimmed.mp4'
    cam1 = downsample(str(cam1))
    cam2 = downsample(str(cam2))
    p1, p2 = extract_pose(cam1), extract_pose(cam2)
    joblib.dump(p1, outpath / 'cam1_pose.pkl')
    joblib.dump(p2, outpath / 'cam2_pose.pkl')
    print(f'Processed {session.name} in {time.time() - t0} seconds')

t00 = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_session, sessions)
print(f'Processed all sessions in {time.time() - t00} seconds')


