# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import cv2
import time
import joblib
import shutil
import numpy as np
import argparse
from loguru import logger
from pathlib import Path

sys.path.append('.')
from psypose.run_pare import PARETester
from psypose.utils import split_tracks
from pare.utils.smooth_pose import smooth_pose
from pare.utils.demo_utils import (
    download_youtube_clip,
    video_to_images,
    images_to_video,
)

import pare
pare_loc = os.path.dirname(pare.__file__)

# How to make this flexible between a command-line thing and a jupyter thing.
# I think I need to make a class that has all the arguments as attributes, and then I can just pass that class to the functions.

CFG = pare_loc+'/data/pare/checkpoints/pare_w_3dpw_config.yaml'
CKPT = pare_loc+'/data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt'
MIN_NUM_FRAMES = 0


parser = argparse.ArgumentParser()

parser.add_argument('--save_vertices', type=bool, default=False,
                    help='save vertices in output file. False reduces size of file.')

parser.add_argument('--cfg', type=str, default=CFG,
                    help='config file that defines model hyperparams')

parser.add_argument('--ckpt', type=str, default=CKPT,
                    help='checkpoint path')

parser.add_argument('--exp', type=str, default='',
                    help='short description of the experiment')

parser.add_argument('--mode', default='video', choices=['video', 'folder', 'webcam'],
                    help='Demo type')

parser.add_argument('--vid_file', type=str,
                    help='input video path or youtube link')

parser.add_argument('--image_folder', type=str,
                    help='input image folder')

parser.add_argument('--output_folder', type=str, default='logs/demo/demo_results',
                    help='output folder to write results')

parser.add_argument('--tracking_method', type=str, default='bbox', choices=['bbox', 'pose'],
                    help='tracking method to calculate the tracklet of a subject from the input video')

parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                    help='object detector to be used for bbox tracking')

parser.add_argument('--yolo_img_size', type=int, default=416,
                    help='input image size for yolo detector')

parser.add_argument('--tracker_batch_size', type=int, default=12,
                    help='batch size of object detector used for bbox tracking')

parser.add_argument('--staf_dir', type=str, default='/home/mkocabas/developments/openposetrack',
                    help='path to directory STAF pose tracking method installed.')

parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size of PARE')

parser.add_argument('--display', type=bool, default=False,
                    help='visualize the results of each step during demo')

parser.add_argument('--smooth', type=bool, default=True,
                    help='smooth the results to prevent jitter')

parser.add_argument('--min_cutoff', type=float, default=0.004,
                    help='one euro filter min cutoff. '
                         'Decreasing the minimum cutoff frequency decreases slow speed jitter')

parser.add_argument('--beta', type=float, default=0.7,
                    help='one euro filter beta. '
                         'Increasing the speed coefficient(beta) decreases speed lag.')

parser.add_argument('--no_render', type=bool, default=True,
                    help='disable final rendering of output video.')

parser.add_argument('--no_save', type=bool, default=True,
                    help='disable final save of output results.')

parser.add_argument('--wireframe', type=bool, default=False,
                    help='render all meshes as wireframes.')

parser.add_argument('--sideview', type=bool, default=False,
                    help='render meshes from alternate viewpoint.')

parser.add_argument('--draw_keypoints', type=bool, default=False,
                    help='draw 2d keypoints on rendered image.')

parser.add_argument('--save_obj', type=bool, default=False,
                    help='save results as .obj files.')

parser.add_argument('--shot_detection', type=bool, default=True,
                    help='Run psypose shot detection with PySceneDetect')

parser.add_argument('--static_cam', type=bool, default=True,
                    help='Whether camera is static or not. If True, multiperson tracker will only track a small number of frames.')

parser.add_argument('--expected_n_people', type=int, default=None,
                    help='Number of people expected to be in the frame. If None, will use the number of people in the first frame.')


args = parser.parse_args(args=[])

def estimate_pose(pose):
    """
    @param args: All of the arguments provided by the PARE developers for their model.
    @param pose: A PsyPose pose object with a vid_path attribute.
    @return: Returns the results of the PARE model.
    """

    # add all args in args to pose object (if not already in pose object)
    for arg in vars(args):
        if arg not in pose.__dict__:
            setattr(pose, arg, getattr(args, arg))

    pose.shot_detection = args.shot_detection
    pose.smooth = args.smooth
    pose.subset_folder = None
    pose.min_cutoff = args.min_cutoff

    demo_mode = args.mode

    pose.image_folder = str(pose.vid_path.parent / os.path.basename(pose.vid_path).split('.')[0]) + '_frames'

    # check if image folder exists. If it does AND it has images in it already:
    # check if n_frames==frames in video. If yes, skip extraction. If no, delete the frames and extract again.

    # *** For some reason the test fails even if the frames are already extracted. Check count stuff.

    parse_to_frames = True
    if os.path.isdir(pose.image_folder):
        n_frames = len(os.listdir(pose.image_folder))
        print('N_FRAMES', n_frames)
        print('FRAMECOUNT', pose.framecount)
        if n_frames == pose.framecount:
            parse_to_frames = False
            print('Frames already extracted. Skipping extraction.')
        else:
            print('Frames already extracted, but number of frames does not match video. Deleting and extracting again.')
            shutil.rmtree(pose.image_folder)
            os.mkdir(pose.image_folder)


    Path(pose.image_folder).mkdir(exist_ok=True)

    if demo_mode == 'video':
        video_file = str(pose.vid_path)

        # ========= [Optional] download the youtube video ========= #
        if video_file.startswith('https://www.youtube.com'):
            logger.info(f'Downloading YouTube video \"{video_file}\"')
            video_file = download_youtube_clip(video_file, '/tmp')

            if video_file is None:
                exit('Youtube url is not valid!')

            logger.info(f'YouTube Video has been downloaded to {video_file}...')

        if not os.path.isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')

        output_path = os.path.join(pose.output_path, os.path.basename(video_file).replace('.mp4', '_data' + args.exp))
        os.makedirs(output_path, exist_ok=True)

        # if os.path.isdir(os.path.join(output_path, 'tmp_images')):
        #     input_image_folder = os.path.join(output_path, 'tmp_images')
        #     logger.info(f'Frames are already extracted in \"{input_image_folder}\"')
        #     num_frames = len(os.listdir(input_image_folder))
        #     img_shape = cv2.imread(os.path.join(input_image_folder, '000001.png')).shape
        # else:
        print('Extracting frames using ffmpeg...')
        input_image_folder, num_frames, img_shape = video_to_images(
            video_file,
            img_folder=pose.image_folder,
            return_info=True
        )

        # static cam mode is for videos where the camera is static and the number of people is constant
        # I use this to reduce the number of frames to track for faster processing
        if pose.static_cam:
            pose.subset_folder = pose.image_folder + '_subset'
            subset_folder = Path(pose.subset_folder)
            subset_folder.mkdir(exist_ok=True)
            # select frames to use for tracking
            select_frames = [pose.framecount // 4, pose.framecount // 2, (pose.framecount // 4) * 3]
            # copy select frames to subset folder
            for frame in select_frames:
                shutil.copy(os.path.join(pose.image_folder, f'{frame:06d}.png'), pose.subset_folder)

        output_img_folder = f'{input_image_folder}_output'
        #os.makedirs(output_img_folder, exist_ok=True)
    elif demo_mode == 'folder':
        args.tracker_batch_size = 1
        input_image_folder = args.image_folder
        output_path = os.path.join(args.output_folder, input_image_folder.rstrip('/').split('/')[-1] + '_' + args.exp)
        os.makedirs(output_path, exist_ok=True)

        output_img_folder = os.path.join(output_path, 'pare_results')
        os.makedirs(output_img_folder, exist_ok=True)

        num_frames = len(os.listdir(input_image_folder))
    elif demo_mode == 'webcam':
        logger.error('Webcam demo is not implemented!..')
        raise NotImplementedError
    else:
        raise ValueError(f'{demo_mode} is not a valid demo mode.')

    logger.add(
        os.path.join(output_path, 'demo.log'),
        level='INFO',
        colorize=False,
    )
    logger.info(f'Demo options: \n {args}')

    tester = PARETester(args)

    total_time = time.time()
    if args.mode == 'video':
        logger.info(f'Input video number of frames {num_frames}')
        orig_height, orig_width = img_shape[:2]
        total_time = time.time()
        # Here static cam is used to reduce the number of frames to track for faster processing
        if pose.static_cam:
            tracking_results = tester.run_tracking(video_file, pose.subset_folder)
            # reformat results to static bbox
            for track in list(tracking_results.keys()):
                avg_bbox = np.mean(tracking_results[track]['bbox'], axis=0)
                # repeat for n_frames
                tracking_results[track]['bbox'] = np.repeat(avg_bbox[np.newaxis,:], num_frames, axis=0)
                tracking_results[track]['frames'] = np.arange(num_frames)

        else:
            tracking_results = tester.run_tracking(video_file, input_image_folder)
            # if args.shot_detection:
            #     print('\n'+'\033[1m'+'Splitting tracks based on shot detection...\n')
            #     tracking_results, pose.num_splits, pose.split_frames = split_tracks(tracking_results, pose.shots)
        pare_time = time.time()
        print('INPUT IMAGE FOLDER', input_image_folder)

        pare_results = tester.run_on_video(tracking_results, input_image_folder, orig_width, orig_height)
        if not args.save_vertices:
            for track in list(pare_results.keys()):
                # removing vertices because we don't want them (too big)
                if pose.no_render:
                    del pare_results[track]['verts']

        end = time.time()

        fps = num_frames / (end - pare_time)

        del tester.model

        logger.info(f'PARE FPS: {fps:.2f}')
        total_time = time.time() - total_time
        logger.info(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
        logger.info(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

        if not args.no_save:
            logger.info(f'Saving output results to \"{os.path.join(output_path, "pare_output.pkl")}\".')
            joblib.dump(pare_results, os.path.join(output_path, "pare_output.pkl"))

        args.no_render = pose.no_render

        if not args.no_render:
            Path(output_path).mkdir(exist_ok=True)
            Path(output_img_folder).mkdir(exist_ok=True)
            tester.render_results(pare_results, input_image_folder, output_img_folder, output_path,
                                  orig_width, orig_height, num_frames)

            # ========= Save rendered video ========= #
            vid_name = os.path.basename(video_file)
            save_name = f'{vid_name.replace(".mp4", "")}_{args.exp}_result.mp4'
            save_name = os.path.join(output_path, save_name)
            logger.info(f'Saving result video to {save_name}')
            images_to_video(img_folder=output_img_folder, output_vid_file=save_name)

            # Save the input video as well
            images_to_video(img_folder=input_image_folder, output_vid_file=os.path.join(output_path, vid_name))
            shutil.rmtree(output_img_folder)

        #shutil.rmtree(input_image_folder)
        #shutil.rmtree(pose.subset_folder)
        if args.save_obj:
            logger.info(f'Saving output results to \"{os.path.join(output_path, "pare_output.pkl")}\".')
            joblib.dump(pare_results, os.path.join(output_path, "pare_output.pkl"))

        # change track ids to start from 0
        count = -1
        for track in list(pare_results.keys()):
            count+=1
            pare_results[count] = pare_results[track]
            del pare_results[track]
        return pare_results
    elif args.mode == 'folder':
        logger.info(f'Number of input frames {num_frames}')

        total_time = time.time()
        detections = tester.run_detector(input_image_folder)
        pare_time = time.time()
        tester.run_on_image_folder(input_image_folder, detections, output_path, output_img_folder)
        end = time.time()

        fps = num_frames / (end - pare_time)

        del tester.model

        logger.info(f'PARE FPS: {fps:.2f}')
        total_time = time.time() - total_time
        logger.info(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
        logger.info(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

        outputs = {
            'pare_results': pare_results,
            'image_folder': input_image_folder,
        }

        return pare_results


    logger.info('================= END =================')




