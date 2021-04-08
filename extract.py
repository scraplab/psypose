#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:19:37 2021

@author: f004swn
"""

"""
Tools for extracting pose features from a pose object.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import networkx as nx
#import matplotlib
import glob
import scipy.stats
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import AgglomerativeClustering
#os.system('cd ..')
from psypose import utils
from psypose.psypose_mpt.multi_person_tracker.mpt import MPT
from pliers.extractors import merge_results, FaceRecognitionFaceLocationsExtractor, FaceRecognitionFaceEncodingsExtractor  
from pliers.stimuli import VideoStim
from pliers.filters import FrameSamplingFilter
from pliers.converters import VideoFrameCollectionIterator
import ast
import sys
import pdb
import os.path as osp
import nibabel as nib
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
sys.path.append(os.getcwd())


# this won't work on macOS
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import shutil
import time
import torch
import colorsys
import yaml
from tqdm import tqdm
#from multi_person_tracker import MPT
from torch.utils.data import TensorDataset, DataLoader

#MEVA specific stuff
from MEVA.meva.lib.meva_model import MEVA, MEVA_demo
from MEVA.meva.utils.renderer import Renderer
from MEVA.meva.utils.kp_utils import convert_kps
from MEVA.meva.dataloaders.inference import Inference
from MEVA.meva.utils.video_config import parse_args, update_cfg
from MEVA.meva.utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)


from psypose.models import facenet_keras, deepface

def annotate(pose, output_path=None, tracking_method='bbox', 
    vibe_batch_size=225, mesh_out=False, run_smplify=False, render=False, wireframe=False,
    sideview=False, display=False, save_obj=False, gpu_id=0, output_folder='MEVA_outputs',
    detector='yolo', yolo_img_size=416, exp='', cfg=''):
        
     # ========= Run shot detection ========= #
     
     shots = utils.get_shots(pose.vid_path)
     # Here, shots is a list of tuples (each tuple contains the in and out frames of each shot)
     
     # ========= Prepare video for pose annotation ========= #
     cv2_array = utils.video_to_array(pose.video_cv2)
 
#     MIN_NUM_FRAMES=1
     
     torch.cuda.set_device(gpu_id)
     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

     filename = os.path.splitext(os.path.basename(pose.vid_path))[0]
     output_path = os.path.join(output_folder, filename)
     os.makedirs(output_path, exist_ok=True)
  
     num_frames, img_shape = pose.n_frames, pose.video_shape # Move this to the loop for each shot
 
     print(f'Input video number of frames {num_frames}')
     orig_height, orig_width = img_shape[:2]
 
     total_time = time.time()
 
     # ========= Run tracking ========= #
     
     # Initialize tracker
     mot = MPT(
         device=device,
         #batch_size=tracker_batch_size,
         batch_size=1,
         display=display,
         detector_type=detector,
         output_format='dict',
         yolo_img_size=yolo_img_size,
     )
     
     print("Running pose estimation shot-by-shot...")
     tracking_results = mot(cv2_array, shots)
         
 
         # # remove tracklets if num_frames is less than MIN_NUM_FRAMES
         # for person_id in list(tracking_results.keys()):
         #     if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
         #         del tracking_results[person_id]
     
 
     # ========= MEVA Model ========= #
     pretrained_file = f"results/meva/{exp}/model_best.pth.tar"
 
     config_file = osp.join("meva/cfg", f"{cfg}.yml")
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
 
     
     ckpt = torch.load(pretrained_file)
     # print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
     ckpt = ckpt['gen_state_dict']
     model.load_state_dict(ckpt)
     model.eval()
     print(f'Loaded pretrained weights from \"{pretrained_file}\"')
 
     
     # ========= Run MEVA on each person ========= #
     bbox_scale = 1.2
     print('Running MEVA on each tracklet...')
     vibe_time = time.time()
     vibe_results = {}
     for person_id in tqdm(list(tracking_results.keys())):
         bboxes = joints2d = None
 
         bboxes = tracking_results[person_id]['bbox']
         frames = tracking_results[person_id]['frames']
         if len(frames) < 90:
             print(f"!!!tracklet < 90 frames: {len(frames)} frames")
             continue
 
         dataset = Inference(
             frames=frames,
             bboxes=bboxes,
             joints2d=joints2d,
             scale=bbox_scale,
         )
 
         bboxes = dataset.bboxes
         frames = dataset.frames
 
         #dataloader = DataLoader(dataset, batch_size=vibe_batch_size, num_workers=16, shuffle = False)
 
         with torch.no_grad():
 
             pred_cam, pred_pose, pred_betas, pred_joints3d = [], [], [], [], []
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
             'joints2d': joints2d,
             'bboxes': bboxes,
             'frame_ids': frames,
         }
 
         vibe_results[person_id] = output_dict
 
     del model
 
 
     end = time.time()
     fps = num_frames / (end - vibe_time)
 
     print(f'VIBE FPS: {fps:.2f}')
     total_time = time.time() - total_time
     print(f'Total time spent: {total_time:.2f} seconds (including model loading time).')
     print(f'Total FPS (including model loading time): {num_frames / total_time:.2f}.')
 
     print(f'Saving output results to \"{os.path.join(output_path, "meva_output.pkl")}\".')
 
     joblib.dump(vibe_results, os.path.join(output_path, "meva_output.pkl"))
     pose.pose_data = vibe_results
 
     print('================= END =================')
 
 
 
 