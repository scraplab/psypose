#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:45:24 2021

@author: f004swn
"""

import os
import sys
import os.path as osp

#sys.path.append('psypose/MEVA')
#sys.path.append('psypose/MEVA/meva')
#sys.path.append('psypose/MEVA/meva/cfg')


os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import numpy as np
import atexit
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader


from psypose.MEVA.meva.lib.meva_model import MEVA_demo
from psypose.MEVA.meva.utils.renderer import Renderer
from psypose.MEVA.meva.dataloaders.inference import Inference
from psypose.MEVA.meva.utils.video_config import update_cfg
from psypose.MEVA.meva.utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    images_to_video
)

from psypose.utils import PSYPOSE_DATA_DIR, video_to_images

out_dir = os.getcwd()
dir_name = osp.dirname(__file__)+'/MEVA/'

def estimate_pose(pose, save_pkl=False, image_folder=out_dir+'/images_intermediate', output_path=None, tracking_method='bbox', 
    vibe_batch_size=225, tracker_batch_size=12, mesh_out=False, run_smplify=False, render=False, wireframe=False,
    sideview=False, display=False, save_obj=False, gpu_id=0, output_folder='MEVA_outputs',
    detector='yolo', yolo_img_size=416, exp='train_meva_2', cfg='train_meva_2'):

    #return_dir = os.getcwd()
    #os.chdir('MEVA')
    
    video_file = pose.vid_path
    
    # setting minimum number of frames to reflect minimum track length to half a second
    MIN_NUM_FRAMES = 25
    #MIN_NUM_FRAMES = round(pose.fps/2)

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')

    filename = os.path.splitext(os.path.basename(video_file))[0]
    #output_path = os.path.join(output_folder, filename)
    #os.makedirs(output_path, exist_ok=True)
 
    image_folder, num_frames, img_shape = video_to_images(video_file, img_folder=image_folder, return_info=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=tracker_batch_size,
        display=display,
        detector_type=detector,
        output_format='dict',
        yolo_img_size=yolo_img_size,
    )
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # print('Track lengths: /n')
    # for person_id in list(tracking_results.keys()):
    #     print(str(tracking_results[person_id]['frames'].shape[0]))
    

    # ========= MEVA Model ========= #
    pretrained_file = PSYPOSE_DATA_DIR.joinpath("meva_data", "model_best.pth.tar")

    config_file = osp.join(dir_name, "meva", "cfg", f"{cfg}.yml")
    cfg = update_cfg(config_file)
    model = MEVA_demo(
        n_layers=cfg.MODEL.TGRU.NUM_LAYERS,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        hidden_size=cfg.MODEL.TGRU.HIDDEN_SIZE,
        add_linear=cfg.MODEL.TGRU.ADD_LINEAR,
        bidirectional=cfg.MODEL.TGRU.BIDIRECTIONAL,
        use_residual=cfg.MODEL.TGRU.RESIDUAL,
        cfg = cfg.VAE_CFG,
    ).to(device)

    
    ckpt = torch.load(pretrained_file, map_location=device)
    # print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    # ========= MEVA Model ========= #

    
    # ========= Run MEVA on each person ========= #
    bbox_scale = 1.2
    print('Running MEVA on each tracklet...')
    vibe_time = time.time()
    meva_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']
    #    if len(frames) < 90:
    #        print(f"!!!tracklet < 90 frames: {len(frames)} frames")
    #        continue

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames

        dataloader = DataLoader(dataset, batch_size=vibe_batch_size, num_workers=16, shuffle = False)

        with torch.no_grad():

            pred_cam, pred_pose, pred_betas, pred_joints3d = [], [], [], []
            data_chunks = dataset.iter_data() 

            for idx in range(len(data_chunks)):
                batch = data_chunks[idx]
                batch_image = batch['batch'].unsqueeze(0)
                cl = batch['cl']
                batch_image = batch_image.to(device)

                batch_size, seqlen = batch_image.shape[:2]
                output = model(batch_image)[-1]

                pred_cam.append(output['theta'][0, cl[0]: cl[1], :3])
                pred_pose.append(output['theta'][0,cl[0]: cl[1],3:75])
                pred_betas.append(output['theta'][0, cl[0]: cl[1],75:])
                pred_joints3d.append(output['kp_3d'][0, cl[0]: cl[1]])


            pred_cam = torch.cat(pred_cam, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch_image


        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        meva_results[person_id] = output_dict

    del model


    end = time.time()
    fps = num_frames / (end - vibe_time)

    print(f'VIBE FPS: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
    print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')

   # if save_pkl:
   #     print(f'Saving output results to \"{os.path.join(output_path, "meva_output.pkl")}\".')
    
   #     joblib.dump(meva_results, os.path.join(output_path, "meva_output.pkl"))

    # meva_results = joblib.load(os.path.join(output_path, "meva_output.pkl"))

    #if render_preview or not len(meva_results) == 0:
    if render:
        # ========= Render results as a single video ========= #
        renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=wireframe)

        output_img_folder = f'{image_folder}_output'
        os.makedirs(output_img_folder, exist_ok=True)

        print(f'Rendering output video, writing frames to {output_img_folder}')

        # prepare results for rendering
        frame_results = prepare_rendering_results(meva_results, num_frames)
        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in meva_results.keys()}

        image_file_names = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ])

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)
            # img = np.zeros(img.shape)

            if sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']

                mc = mesh_color[person_id]

                mesh_filename = None

                if save_obj:
                    mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    mesh_filename=mesh_filename,
                )
                
                frame_cam = np.array([ 0.5,  1., 0,  0])
                if sideview:
                    side_img = renderer.render(
                        side_img,
                        frame_verts,
                        cam=frame_cam,
                        color=mc,
                        mesh_filename=mesh_filename,
                        # angle=270,
                        # axis=[0,1,0],
                    )

            if sideview:
                img = np.concatenate([img, side_img], axis=1)

            cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.png'), img)

            if display:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        vid_name = os.path.basename(video_file)
        save_name = f'{vid_name.replace(".mp4", "")}_meva_result.mp4'
        save_name = os.path.join(output_path, save_name)
        print(f'Saving result video to {save_name}')
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        shutil.rmtree(output_img_folder)

    def clean_image_folder():
        if osp.exists(image_folder) and osp.isdir(image_folder):
            shutil.rmtree(image_folder)

    atexit.register(clean_image_folder)
    os.chdir(return_dir)


    print('========FINISHED POSE ESTIMATION========')
    return meva_results




