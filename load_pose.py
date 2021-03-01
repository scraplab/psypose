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
from pliers.extractors import merge_results, FaceRecognitionFaceLocationsExtractor, FaceRecognitionFaceEncodingsExtractor  
from pliers.stimuli import VideoStim
from pliers.filters import FrameSamplingFilter
from pliers.converters import VideoFrameCollectionIterator


class pose(object):
    
    def __init__(self):
        self.is_clustered = False
        self.clusters_named = False
        pass
    
            
    def load_face_data(self, face_data):
        self.face_data_path = os.path.abspath(face_data)
        global face_data_opened
        face_data_opened = pd.read_csv(face_data,
                               converters = {'location':utils.string2list, 'encodings':utils.from_np_array})
        self.face_data = face_data_opened
        
    def load_video(self, vid_path):
        vid_path = os.path.abspath(vid_path)
        self.video_cv2 = cv2.VideoCapture(vid_path)
        self.fps = self.video_cv2.get(cv2.CAP_PROP_FPS)
        self.vid_path = vid_path
        self.n_frames = int(self.video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
        
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
        
        # if overwrite:
        #     self.clusters=None
        
        
        # if (self.is_clustered & overwrite==False):
        #     raise Exception('Pose object has already been clustered. Set overwrite=True to overwite previous clustering.')
        
        face_data = self.face_data
        vibe_data = self.pose_data

        fr_ids = []  
        # here, iterating through all rows of the face_rec data frame (all available frames)
        # checking if each frame is listed within any of the VIBE tracks
        # if overlap frames are detected, the algo will check if the face box is within 
        # the VIBE bounding box. If True, then that frame will get that track ID  
        for r in range(len(face_data)):
            row = face_data.iloc[r]
            frame = row['frameID']
            track_id_list = []
            for t in np.unique(list(vibe_data.keys())):
                track = vibe_data.get(t)
                track_frames = track.get('frame_ids')
                if frame in track_frames:
                    track_id_list.append(t)
            for track_id in track_id_list:
                box_loc = np.where(vibe_data.get(track_id).get('frame_ids')==frame)[0][0]
                box = vibe_data.get(track_id).get('bboxes')[box_loc]
                if utils.check_match(box, row['location']):
                    track_designation = int(track_id)
                    break
                else:
                    # track designation is 0 if the face is not within body box,
                    # those rows will be removed.
                    track_designation = 0
            fr_ids.append(track_designation)
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
            frames_1 = vibe_data.get(track1).get('frame_ids')
            for j, track2 in enumerate(opt_tracks):
                frames_2 = vibe_data.get(track2).get('frame_ids')
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
    
    def annotate(self, output_path=None, image_folder=None, tracking_method='bbox', 
    vibe_batch_size=225, mesh_out=False, run_smplify=False, render=False, wireframe=False,
    sideview=False, display=False, save_obj=False, gpu_id=0):
        from VIBE.annotate import annotate
        annotate(self, output_path=output_path, image_folder=image_folder, tracking_method=tracking_method, 
                 vibe_batch_size=vibe_batch_size, mesh_out=mesh_out, run_smplify=run_smplify, render=render, wireframe=wireframe,
                 sideview=sideview, display=display, save_obj=save_obj, gpu_id=gpu_id)
        
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
                track_frames = self.vibe_data.get(track).get('frame_ids')
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
            # 
            enc = trans[parameter]
            arr = np.zeros((enc.shape[0], len(enc[0])))
            for frame in range(arr.shape[0]):
                arr[frame,:]=enc[frame]
            return arr
    
        vid = VideoStim(self.vid_path)
        vid_filt = FrameSamplingFilter(every=every).transform(vid)
        frame_conv = VideoFrameCollectionIterator()
        vid_frames = frame_conv.transform(vid_filt)
        
        ext_loc = FaceRecognitionFaceLocationsExtractor()
        #for this, change to model='cnn' for GPU use on discovery

        print("Extracting face locations")
        # This is using pliers, which uses dlib for extraction face locations (SOTA)
        face_locs_data = ext_loc.transform(vid_frames)
        #remove frames without faces
        face_locs_data = [obj for obj in face_locs_data if len(obj.raw) != 0]
        # convert to dataframe and remove frames with no faces
        face_df = pd.DataFrame(dict(zip(['frame_ids', 'locations'],
                                               [[obj.stim.frame_num for obj in face_locs_data],
                                                [obj.raw for obj in face_locs_data if len(obj.raw) != 0]])))
        face_imgs = [utils.crop_image(obj.stim.data, obj.raw[0]) for obj in face_locs_data]
        
        print("Extracting face encodings")
        if encoder=='default':
            encoder = utils.default_encoding
        elif encoder=='facenet':
            from psypose.models import facenet_keras
            encoder = facenet_keras.encode
        elif encoder=='deepface':
            from psypose.models import deepface
            encoder = deepface.encode
        
        encodings = [encoder(image) for image in face_imgs]
        face_df['encodings'] = encodings
        
        self.face_data_opened = face_df
        if out != None:
            print('Saving results...')
            face_df.to_csv(out)
    
    ## add face locations and face encodings as separate self properties?

        
    # def consolidate_clusters(self):
    #     tot_frames = int(self.video_cv2.get(cv2.CAP_PROP_FRAME_COUNT))
    
