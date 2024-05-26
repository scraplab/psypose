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

workdir = Path('/mnt/7db0cf53-29c3-4419-ad97-e183309eb002/landry_dev_scratch/2023-10-24_000/subclip_900_920')
# get all the cams
vpaths = list(workdir.glob('*.mp4'))
print(vpaths)

def extract_pose(vpath, render=False):
    if not isinstance(vpath, Path):
        vpath = Path(vpath)
    pose = data.pose()
    pose.load_video(vpath)
    pose.static_cam = True
    pose.no_render = not render
    estimate = pose_estimation.estimate_pose(pose)
    return estimate

def get_basename(vpath):
    return os.path.basename(vpath).split('.')[0]

for vpath in vpaths:
    estimate = extract_pose(vpath)
    opath = vpath.with_name(vpath.stem + '_pose.pkl')
    joblib.dump(estimate, opath)
