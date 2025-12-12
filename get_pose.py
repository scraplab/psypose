#!/usr/bin/env python3

import argparse
from pathlib import Path
import joblib
import moviepy.editor as mp
from psypose import data, pose_estimation

def extract_pose(vpath):
    """Extract pose from a video file."""
    if not isinstance(vpath, Path):
        vpath = Path(vpath)
    
    # Set up output path
    opath = vpath.with_name(vpath.stem + '_pose.pkl')
    
    # Extract pose
    pose = data.Camera()
    pose.output_path = opath
    pose.load_video(vpath)
    pose.no_render = True
    estimate = pose_estimation.estimate_pose(pose)
    
    # Save the pose data
    joblib.dump(estimate, opath)
    print(f'Pose data saved to: {opath}')
    return estimate

def main():
    parser = argparse.ArgumentParser(description='Extract pose data from a video file')
    parser.add_argument('-i', '--input', required=True, help='Path to input video file')
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist")
        return
    
    try:
        extract_pose(args.input)
        print('Pose extraction completed successfully!')
    except Exception as e:
        print(f'Error during pose extraction: {e}')

if __name__ == '__main__':
    main() 