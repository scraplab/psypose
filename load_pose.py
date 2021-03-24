#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 13:50:03 2021

@author: f004swn
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
from psypose.multi_person_tracker.multi_person_tracker.mpt import MPT
from pliers.extractors import merge_results, FaceRecognitionFaceLocationsExtractor, FaceRecognitionFaceEncodingsExtractor  
from pliers.stimuli import VideoStim
from pliers.filters import FrameSamplingFilter
from pliers.converters import VideoFrameCollectionIterator
import ast
import sys
import pdb
import os.path as osp
import nibabel as nib
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
class pose(object):
    
    def __init__(self):
        self.is_clustered = False
        self.clusters_named = False
        pass
    
    def load_fmri(self, fmri_path, TR):
        self.fmri_path = fmri_path
        self.brain_data = nib.load(fmri_path).get_fdata()
        self.TR = TR
        # all timeseries will be in milliseconds for accuracy.
        self.brain_time = [1000*TR for tr in range(self.brain_data.shape[0])]
        
    def load_face_data(self, face_data):
        self.face_data_path = os.path.abspath(face_data)
        
        face_data_array = np.genfromtxt(face_data, delimiter=',')
        n_frames = face_data_array.shape[0]
        face_df = pd.DataFrame(columns=['frame_ids', 'locations', 'encodings'])
        face_df['frame_ids'] = face_data_array[:,0]
        face_df['locations'] = [face_data_array[i,1:5] for i in range(n_frames)]
        face_df['encodings'] = [face_data_array[i,5:] for i in range(n_frames)]
        self.face_data = face_df
        
    def load_video(self, vid_path):
        vid_path = os.path.abspath(vid_path)
        self.video_cv2 = cv2.VideoCapture(vid_path)
        self.fps = self.video_cv2.get(cv2.CAP_PROP_FPS)
        self.vid_path = vid_path
        self.framecount = int(self.video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_time = [(1/self.fps)*1000*frame for frame in range(self.framecount)]
        self.video_shape = (int(self.video_cv2.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cv2.get(cv2.CAP_PROP_FRAME_WIDTH)))
        
    def load_pkl(self, pkl_path):
        self.pkl_path = pkl_path
        global pkl_open
        pkl_open = dict(joblib.load(pkl_path))
        self.pose_data = pkl_open
        self.n_tracks = len(pkl_open)


    def cluster_ID(self, metric='cosine', linkage='average', overwrite=False, use_cooccurence=True):
        
        """
        Clusters all tracks based on facial encodings attached to each track. 
        It is recommended to use the cosine metric and average linkage for best results.
        Outputs a dictionary where keys are cluster ID's and values are lists of tracks.
        
        """
        
        if overwrite:
            self.clusters=None
        
        if self.is_clustered and not overwrite:
            raise Exception('Pose object has already been clustered. Set overwrite=True to overwite previous clustering.')
        
        face_data = self.face_data
        pose_data = self.pose_data

        fr_ids = []  
        # here, iterating through all rows of the face_rec data frame (all available frames)
        # checking if each frame is listed within any of the VIBE tracks
        # if overlap frames are detected, the algo will check if the face box is within 
        # the VIBE bounding box. If True, then that frame will get that track ID  
        for r in range(len(face_data)):
            row = face_data.iloc[r]
            frame = row['frame_ids']
            track_id_list = []
            for t in np.unique(list(pose_data.keys())):
                track = pose_data.get(t)
                track_frames = track.get('frame_ids')
                if frame in track_frames:
                    track_id_list.append(t)
            if len(track_id_list)!=0:
                for track_id in track_id_list:
                    box_loc = np.where(pose_data.get(track_id).get('frame_ids')==frame)[0][0]
                    box = pose_data.get(track_id).get('bboxes')[box_loc]
                    if utils.check_match(box, row['locations']):
                        track_designation = int(track_id)
                        break
                    else:
                        # track designation is 0 if the face is not within body box,
                        # those rows will be removed.
                        track_designation = 0
                fr_ids.append(track_designation)
            else:
                fr_ids.append(0)
        face_data['track_id'] = fr_ids
        #removing face encodings with no match
        face_data = face_data[face_data['track_id']!=0]
        
        # here, I'm going through each unique track and getting an average face encoding
        avg_encodings = []
        enc_tracks = []
        for track in [int(i) for i in np.unique(face_data['track_id'])]:
            tr_df = face_data[face_data['track_id']==track]
            avg_encodings.append(np.mean(tr_df['encodings']))
            enc_tracks.append(track)
            
        track_enc_avgs = np.array(avg_encodings)
        track_encoding_avgs = dict(zip(enc_tracks, avg_encodings))
        # here, we cluster all of the averaged track encodings. 
        # tracks containing the same face will be concatenated later
        
        def pd_to_arr(pand_series):
            # converts multi-dimensional pandas series to np array
            x = len(pand_series)
            y = len(pand_series.iloc[0])
            arr = np.zeros((x,y))
            for i in range(x):
                arr[i] = pand_series.iloc[i]
            return arr
        
        
        # Here, I'm going to compute the between-track distances using only co-occuring tracks
        # I'll first create a co-occurence matrix
        n_tracks = len(np.unique(face_data['track_id']))
        
        # Getting a list of track ID's with faces in the,. 
    
        opt_tracks = np.unique(face_data['track_id']).tolist()
        # Initialize empty co-occurence matrix
        cooc = np.zeros((n_tracks, n_tracks))
        for i, track1 in enumerate(opt_tracks):
            frames_1 = pose_data.get(track1).get('frame_ids')
            for j, track2 in enumerate(opt_tracks):
                frames_2 = pose_data.get(track2).get('frame_ids')
                if len(np.intersect1d(frames_1, frames_2))>0:
                    cooc[i][j]=True
        cooc = cooc[np.tril_indices_from(cooc, k=-1)]
        #get per-track average encodings with nested list comprehension (small python flex)
        track_encoding_avgs = [np.mean(enc, axis=0) for enc in
                                      [face_data[face_data['track_id']==track]['encodings'].to_numpy()
                                       for track in opt_tracks]]
        encoding_dist_mat = squareform(pdist(track_encoding_avgs, metric=metric))
        all_track_dist = encoding_dist_mat[np.tril_indices_from(encoding_dist_mat, k=-1)]
        inter_track_distances = np.extract(cooc, all_track_dist)
        # get the within-track distances
        # for each track, a distance matrix of all encodings is made
        # the average distances is taken
        intra_track_encodings = [pd_to_arr(face_data[face_data['track_id']==tr]['encodings']) for tr in opt_tracks]
        intra_track_distances = np.array([np.mean(pdist(encs, metric=metric)) for encs in intra_track_encodings])
        intra_track_distances = intra_track_distances[~np.isnan(intra_track_distances)]

        all_distances = np.concatenate((inter_track_distances, intra_track_distances))

        all_distances = all_distances.reshape(-1,1)
        #dist_cluster = AgglomerativeClustering(linkage='ward', distance_threshold=0, n_clusters=None).fit(all_distances)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2).fit(all_distances)
        #clustered = kmeans.fit_transform(all_distances)
        clus0 = all_distances[np.where(kmeans.labels_==0)]
        clus1 = all_distances[np.where(kmeans.labels_==1)]
        cut = float(max(clus0) + (min(clus1)-max(clus0))/2)
        
        # This final clustering uses the cut as a distance threshold. 
        # Tracks in the same cluster will now be consolidated. 
        final_clust = AgglomerativeClustering(linkage=linkage, distance_threshold=cut, 
                                              n_clusters=None, affinity=metric).fit(track_enc_avgs)
        
        clusters = []
        cluster_tracks = []
        enc_tracks = np.array(enc_tracks)
        for clus in np.unique(final_clust.labels_):
            clusters.append(clus)
            tracks = enc_tracks[np.where(final_clust.labels_==clus)[0]]
            cluster_tracks.append(tracks)
        fc = dict(zip(clusters, cluster_tracks))
        cluster_mains = [k for k in sorted(fc, key=lambda k: 
                                           len(fc[k]), reverse=True)]
        t = [fc.get(clust) for clust in cluster_mains]
        sorted_dict = dict(zip([i for i in range(0, len(t))], t))
        
        self.clusters = sorted_dict
        self.is_clustered = True
            
    def name_clusters(self, character_dict, overwrite_names=False):
        # if (self.clusters_named & overwrite_names==False):
        #     raise Exception('Clusters have already been named. Set overwrite=True to overwrite previous labels.')
        # if self.is_named & overwrite_names:
        #     self.named_clusters=None
        #     self.character_key=None
        
        #self.clusters_named = True
        # this function should have the capability of 
        # both extracting main character clusters as well as 
        # merging clusters with the same name 
        
        self.character_key = character_dict
        chars = list(np.unique(character_dict.values())[0])
        names_clustered = {}
        for char in chars:
            common_clusters = [k for k, v in character_dict.items() if str(v) == char]
            tracks = []
            for cluster in common_clusters:
                cluster_tracks = self.clusters.get(cluster)
                tracks.extend(list(cluster_tracks))
            names_clustered.update({char:tracks})
        names_sorted = sorted(names_clustered, key=lambda k: len(names_clustered[k]), reverse=True)
        tracks_sorted = [names_clustered.get(name) for name in names_sorted]
        names_clustered = dict(zip(names_sorted, tracks_sorted))
        self.named_clusters = names_clustered
        
    def imaging_pars(self, functional = 'a_func_file', TR=2.0):
        #make a function that allows you to either manually input imaging parameters
        #or provide a sample functional run for parameters (HRF, TR, etc)
        #this info will be used to generate the different regressors
        self.TR = TR
        
    def presence_matrix(self, character_order, hertz=None):
    
        # character order should be an iterable containing strings of character IDs
        # that correspond to the labels given in name_clusters()
        char_order = np.array(character_order)
    
        # first make an empty array for character appearances
        char_auto = np.zeros((self.n_frames, len(char_order)))
        # this is going to label a character presence array with ones and zeros
        for i, tracklist in enumerate(list(self.named_clusters.values())):
            character = list(self.named_clusters.keys())[i]
            if character not in char_order:
                continue
            arr_placement = int(np.where(char_order==character)[0]) 
            for track in tracklist:
                track_frames = self.pose_data.get(track).get('frame_ids')
                char_auto[:,arr_placement][track_frames] = 1
        char_frame = pd.DataFrame(char_auto, columns=character_order)
        char_frame['frame_ids'] = np.arange(self.n_frames)
        self.full_ID_annotations = char_frame
        if hertz==None:
            return char_frame
        else:
            needed_frames = np.arange(round((1/hertz)*self.fps), self.n_frames, round((1/hertz)*self.fps))
            auto_appearances_filt = char_frame.take(needed_frames[:-2], axis=0).reset_index(drop=True)
            return auto_appearances_filt
        
    def encode_faces(self, overwrite=False, encoder='default', use_TR=False, every=None, out=None):
        if every==None:
            # encode every frame if sampling parameter not defined
            every=1
            
        def get_data(trans, parameter):
            enc = trans[parameter]
            arr = np.zeros((enc.shape[0], len(enc[0])))
            for frame in range(arr.shape[0]):
                arr[frame,:]=enc[frame]
            return arr
        print('\nParsing video data...')
        vid = VideoStim(self.vid_path)
        vid_filt = FrameSamplingFilter(every=every).transform(vid)
        frame_conv = VideoFrameCollectionIterator()
        vid_frames = frame_conv.transform(vid_filt)
        
        ext_loc = FaceRecognitionFaceLocationsExtractor()
        #for this, change to model='cnn' for GPU use on discovery

        print("\nExtracting face locations...")
        # This is using pliers, which uses dlib for extraction face locations (SOTA)
        face_locs_data = ext_loc.transform(vid_frames)
        #remove frames without faces
        face_locs_data = [obj for obj in face_locs_data if len(obj.raw) != 0]
        # convert to dataframe and remove frames with no faces

        frame_ids = [obj.stim.frame_num for obj in face_locs_data]
        locations = [obj.raw for obj in face_locs_data if len(obj.raw) != 0]
        
        # expanding multi-face frames
        frame_ids_expanded = []
        bboxes_expanded = []
        face_imgs = []
        for f, frame in enumerate(frame_ids):
            face_bboxes = locations[f]
            frame_img = utils.frame2array(frame, self.video_cv2)
            for i in face_bboxes:
                frame_ids_expanded.append(frame)
                bboxes_expanded.append(i)
                face_imgs.append(utils.crop_image(frame_img, i))
                
        bboxes = np.array(bboxes_expanded)

        print("\nExtracting face encodings...")
        
        if encoder=='default':
            encoding_length = 128
            encode = utils.default_encoding
        elif encoder=='facenet':
            encoding_length = 128
            encode = facenet_keras.encode
        elif encoder=='deepface':
            encoding_length = 2622
            encode = deepface.encode
        
        out_array = np.empty((len(frame_ids_expanded), 5+encoding_length))
        encodings = []
        for image in tqdm(face_imgs):
            encodings.append(encode(image))
        out_array[:,0], out_array[:,1:5], out_array[:, 5:] = frame_ids_expanded, bboxes, np.array(encodings)
        
        face_df = pd.DataFrame(columns=['frame_ids', 'bboxes', 'encodings'])
        face_df['frame_ids'] = frame_ids_expanded
        face_df['bboxes'] = bboxes_expanded
        face_df['encodings'] = encodings
        
        self.face_data = face_df
        
        if out != None:
            print('Saving results...')
            np.savetxt(out, out_array, delimiter=',')
        self.faces_encoded = True
    
    ## add face locations and face encodings as separate self properties?

        
    # def consolidate_clusters(self):
    #     tot_frames = int(self.video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def annotate_pose(self, output_path=None, tracking_method='bbox', 
    vibe_batch_size=225, mesh_out=False, run_smplify=False, render=False, wireframe=False,
    sideview=False, display=False, save_obj=False, gpu_id=0, output_folder='MEVA_outputs', tracker_batch_size=12,
    detector='yolo', yolo_img_size=416, exp='', cfg=''):
        
        # ========= Run shot detection ========= #
        
        shots = utils.get_shots(self.vid_path)
        # Here, shots is a list of tuples (each tuple contains the in and out frames of each shot)
        
        # ========= Prepare video for pose annotation ========= #
        
        cv2_array = utils.video_to_array(self.video_cv2)
        
        MIN_NUM_FRAMES=25
        
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
        video_file = self.vid_path
    
        if not os.path.isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')
    
        filename = os.path.splitext(os.path.basename(video_file))[0]
        output_path = os.path.join(output_folder, filename)
        os.makedirs(output_path, exist_ok=True)
     
        num_frames, img_shape = self.n_frames, self.video_shape
    
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
        
        
        tracking_results = mot(cv2_array)
    
        # remove tracklets if num_frames is less than MIN_NUM_FRAMES
        for person_id in list(tracking_results.keys()):
            if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
                del tracking_results[person_id]
        
    
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
                image_folder=image_folder,
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
        self.pose_data = vibe_results
    
        shutil.rmtree(image_folder)
        print('================= END =================')
            

    
