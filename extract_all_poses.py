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
import glob

data_dir = Path('/safestore/users/landry/SCRAP/data/andromeda_storage/conversations_unconstrained')
#data_dir = Path('/media/landry/fastscratch/landry_dev_scratch')

pose_names = ['cam1_pose.pkl', 'cam2_pose.pkl']

# get all the sessions in the data directory
def screen_paths(dir):
    removes = []

    sessions = [f for f in dir.iterdir() if f.is_dir()]
    sessions = [i for i in sessions if '_' in i.name]
    sessions = [i for i in sessions if len(i.name.split('_')[1]) == 3]
    sessions = [i for i in sessions if '_remove' not in str(i)]

    for session in sessions:
        if not (session / 'processed').exists():
            os.makedirs(session / 'processed', exist_ok=True)

    # check if the session has the pose files. If it has two pose files, remove it from the list
    for session in sessions:
        processed = str(session / 'processed')
        poses = glob.glob(processed + '/**_pose.pkl')
        poses = [Path(p) for p in poses]
        poses = [p for p in poses if p.exists()]
        if len(poses) == 2:
            print(f'Removing {session.name} because it already has {len(poses)} pose files')
            removes.append(session)
            sessions.remove(session)


    # check if sessions has the concatenated *and* trimmed videos
    for session in sessions:
        videos = [session / 'derivatives' / f for f in ['cam1_concatenated_trimmed.mp4', 'cam2_concatenated_trimmed.mp4']]
        if not all([video.exists() for video in videos]):
            removes.append(session)
            sessions.remove(session)

    print(f'Removed {len(removes)} sessions that did not meet the criteria:')
    for i in removes:
        print(str(i))
        
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
    pose.output_path = opath
    pose.load_video(vpath)
    pose.no_render = True
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
        cam1 = downsample(str(cam1))
        cam2 = downsample(str(cam2))
        try:
            p1, p2 = extract_pose(cam1), extract_pose(cam2)
        except Exception as e:
            print(f'Error processing {session.name}: {e}')
        joblib.dump(p1, outpath / 'cam1_pose.pkl')
        joblib.dump(p2, outpath / 'cam2_pose.pkl')
        print(f'Processed {session.name} in {time.time() - t0} seconds')

# vpath = '/media/landry/fastscratch/landry_dev_scratch/test_outputs/cam1_concatenated_trimmed_trimmed_test.mp4'
# ds = downsample(vpath)

# d = extract_pose(ds)
# joblib.dump(d, '/media/landry/fastscratch/landry_dev_scratch/test_outputs/df540_cam1.pkl')

###

# vpath = '/mnt/7db0cf53-29c3-4419-ad97-e183309eb002/landry_dev_scratch/2023-10-06_000/derivatives/cam2_concatenated_trimmed_ds540.mp4'
# pose = data.pose()
# pose.load_video(vpath)
# pose.no_render = True
# estimate = pose_estimation.estimate_pose(pose)
# joblib.dump(estimate, '/mnt/7db0cf53-29c3-4419-ad97-e183309eb002/landry_dev_scratch/2023-10-06_000/processed/cam2_pose.pkl')


for i in data_dir.iterdir():
    if i.is_dir():
        if (i / 'processed').exists():
            posefiles = [i / 'processed' / f for f in pose_names]
            posefiles = [f for f in posefiles if f.exists()]
            if len(posefiles) == 2:
                print(f'{i.name} has both pose files')
