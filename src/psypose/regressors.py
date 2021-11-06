#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Functions for the construction of second-order pose data and regressors

"""
import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial.distance import pdist, squareform, cdist, euclidean
from scipy.signal import cwt,  morlet2
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.transform import Rotation as R
from psypose import utils
import warnings
import os
from pyquaternion import Quaternion

def cluster_ID(pose, metric='cosine', linkage='average', overwrite=False, use_cooccurence=True):
    
    """
    Clusters all tracks based on facial encodings attached to each track. 
    It is recommended to use the cosine metric and average linkage for best results.
    Outputs a dictionary where keys are cluster ID's and values are lists of tracks.
    
    """
    
    if overwrite:
        pose.clusters=None
    
    if pose.is_clustered and not overwrite:
        raise Exception('Pose object has already been clustered. Set overwrite=True to overwite previous clustering.')
    
    face_data = pose.face_data
    pose_data = pose.pose_data

    pose.encoding_length = len([i for i in list(face_data.columns) if 'enc' in i])

    fr_ids = []  
    # here, iterating through all rows of the face_rec data frame (all available frames)
    # checking if each frame is listed within any of the VIBE tracks
    # if overlap frames are detected, the algo will check if the face box is within 
    # the MEVA bounding box. If True, then that frame will get that track ID  
    for r in range(len(face_data)):
        row = face_data.iloc[r]
        frame = int(row['frame'])
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
                if utils.check_match(box, utils.get_bbox(row)):
                    track_designation = int(track_id)
                    break
                else:
                    # track designation is 'no_match' if the face is not within body box,
                    # those rows will be removed.
                    track_designation = 'no_match'
            fr_ids.append(track_designation)
        else:
            fr_ids.append('no_match')
    face_data['track_id'] = fr_ids
    pose.face_data = face_data
    #removing face encodings with no match
    face_data = face_data[face_data['track_id']!='no_match']
    
    # here, I'm going through each unique track and getting an average face encoding
    enc_columns = [i for i in face_data.columns if 'enc' in i]
    avg_encodings = []
    enc_tracks = []
    for loc, track in enumerate([int(i) for i in np.unique(face_data['track_id'])]):
        tr_df = face_data[face_data['track_id']==track]
        avg_encodings.append(np.mean(tr_df[enc_columns].to_numpy(), axis=0))
        enc_tracks.append(track)
        
    track_enc_avgs = np.array(avg_encodings)
    #print(track_enc_avgs.shape)
    track_encoding_avgs = dict(zip(enc_tracks, avg_encodings))
    # here, we cluster all of the averaged track encodings. 
    # tracks containing the same face will be concatenated later
    
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
                                  [face_data[face_data['track_id']==track][enc_columns].to_numpy()
                                    for track in opt_tracks]]

    encoding_averages = dict(zip(opt_tracks, track_encoding_avgs))
    pose.track_encoding_avgs = encoding_averages
    avg_enc_array = np.empty((len(encoding_averages), pose.encoding_length))
    for i, track in enumerate(list(encoding_averages.keys())):
      avg_enc_array[i] = encoding_averages[track]

    pose.average_encoding_array = avg_enc_array



    encoding_dist_mat = squareform(pdist(track_encoding_avgs, metric=metric))
    all_track_dist = encoding_dist_mat[np.tril_indices_from(encoding_dist_mat, k=-1)]
    inter_track_distances = np.extract(cooc, all_track_dist)
    # get the within-track distances
    # for each track, a distance matrix of all encodings is made
    # the average distances is taken
    intra_track_encodings = [face_data[face_data['track_id']==tr][enc_columns] for tr in opt_tracks]
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

    pose.face_clustering = final_clust
    
    clusters = []
    cluster_tracks = []
    enc_tracks = np.array(enc_tracks)
    for clus in np.unique(final_clust.labels_):
        clusters.append(clus)
        tracks = enc_tracks[np.where(final_clust.labels_==clus)[0]]
        cluster_tracks.append(tracks)
    fc = dict(zip(clusters, cluster_tracks))
    cluster_mains = [k for k in sorted(fc, key=lambda k: len(fc[k]), reverse=True)]
    t = [fc.get(clust) for clust in cluster_mains]
    sorted_dict = dict(zip([i for i in range(0, len(t))], t))
    
    # add cluster ids to face_data
    pose.clusters = sorted_dict
    pose.is_clustered = True
        
def name_clusters(pose, character_dict, overwrite_names=False):
    pose.character_key = character_dict
    chars = list(np.unique(character_dict.keys())[0])
    names_clustered = {}
    for char in chars:
        #common_clusters = [k for k, v in character_dict.items() if str(v) == char]
        tracks = []
        for cluster in character_dict[char]:
            cluster_tracks = pose.clusters[cluster]
            tracks.extend(cluster_tracks)
        names_clustered.update({char:tracks})
    names_sorted = sorted(names_clustered, key=lambda k: len(names_clustered[k]), reverse=True)
    tracks_sorted = [names_clustered[name] for name in names_sorted]
    names_clustered = dict(zip(names_sorted, tracks_sorted))
    pose.named_clusters = names_clustered
    
def imaging_pars(pose, functional = 'a_func_file', TR=2.0):
    #make a function that allows you to either manually input imaging parameters
    #or provide a sample functional run for parameters (HRF, TR, etc)
    #this info will be used to generate the different regressors
    pose.TR = TR
    
def presence_matrix(pose, character_order, hertz=None):

    # character order should be an iterable containing strings of character IDs
    # that correspond to the labels given in name_clusters()
    char_order = np.array(character_order)

    # first make an empty array for character appearances
    char_auto = np.zeros((pose.n_frames, len(char_order)))
    # this is going to label a character presence array with ones and zeros
    for i, tracklist in enumerate(list(pose.named_clusters.values())):
        character = list(pose.named_clusters.keys())[i]
        if character not in char_order:
            continue
        arr_placement = int(np.where(char_order==character)[0]) 
        for track in tracklist:
            track_frames = pose.pose_data.get(track).get('frame_ids')
            char_auto[:,arr_placement][track_frames] = 1
    char_frame = pd.DataFrame(char_auto, columns=character_order)
    char_frame['frame_ids'] = np.arange(pose.n_frames)
    pose.full_ID_annotations = char_frame
    if hertz==None:
        return char_frame
    else:
        needed_frames = np.arange(round((1/hertz)*pose.fps), pose.n_frames, round((1/hertz)*pose.fps))
        auto_appearances_filt = char_frame.take(needed_frames[:-2], axis=0).reset_index(drop=True)
        return auto_appearances_filt

def get_pose_distance(p1, p2):
    # p1 and p2 should just be the vectors
    # may be the normal static vectors or those derived with numpy.gradient()
    p1, p2 = p1[3:], p2[3:]
    return euclidean(p1, p2)

def get_pose_distance(p1, p2):
    # p1 and p2 should just be the vectors
    # may be the normal static vectors or those derived with numpy.gradient()
    p1, p2 = p1[3:], p2[3:]
    return euclidean(p1, p2)

def synchrony(pose, type='static'):
    track_occurence = {}
    data = pose.pose_data
    for frame in range(pose.framecount):
        present_tracks = []
        for key, val in data.items():
            if frame in val['frame_ids']:
                present_tracks.append(key)
        track_occurence[frame] = present_tracks
    pose.track_occurence = track_occurence
    if type=='static':
        max_distance = 26.096028503877093
    elif type=='dynamic':
        max_distance = 52.19205700775419
    out_vec=[]
    for frame in range(pose.framecount):
        tracks = track_occurence[frame]
        if len(tracks) < 2:
            out_vec.append(np.nan)
        else:
            pose_vectors = []
            for track in tracks:
                track_data = data[track]
                if type=='dynamic':
                    track_data['pose_gradient'] = np.gradient(track_data['pose'], axis=0)
                    pose_vectors.append(track_data['pose_gradient'][np.where(track_data['frame_ids']==frame)[0][0]])
                elif type=='static':
                    pose_vectors.append(track_data['pose'][np.where(track_data['frame_ids']==frame)[0][0]])
            sync_arr = np.empty((len(pose_vectors), len(pose_vectors)))
            for i in range(len(pose_vectors)):
                for j in range(len(pose_vectors)):
                    sync_arr[i][j] = get_pose_distance(pose_vectors[i], pose_vectors[j])
            val = np.mean(sync_arr[np.tril_indices_from(sync_arr, k=-1)])
            val = -(2*(val/max_distance)) + 1
            out_vec.append(val)
    return np.array(out_vec)

def flip_quat(quat_arr):
    return quat_arr[np.array([3, 0, 1, 2])]

def add_quat(pose_dat):
    for track, data in pose_dat.items():
        n_frames = len(data['pose'])
        quats = np.empty((n_frames,24,4))
        for i, pose_vec in enumerate(data['pose']):
            pose_vec = pose_vec.reshape((24,3))
            rotvec= R.from_rotvec(pose_vec).as_quat()
            for v, vec in enumerate(rotvec):
                rotvec[v] = flip_quat(vec)
            quats[i] = rotvec
        pose_dat[track].update({'quaternion':quats})

def series_to_wavelets(data, num_windows=10, min_window_width=10, max_window_width=90):
    widths = np.linspace(min_window_width, max_window_width, num_windows)
    wavelets = np.abs(cwt(data, morlet2, widths))
    return wavelets

def pose_to_wavelet_matrix(quaternion_matrix, num_windows=10, min_window_width=10, max_window_width=90):
    n_frames = int(quaternion_matrix.shape[0])
    pose_power_spectrum = np.empty((num_windows*19*4, n_frames))
    idx = 0
    for joint in range(1,20):
        for quat in range(4):
            timeseries = quaternion_matrix[:,joint,quat]
            if quat==3:
                theta_mean = np.mean(timeseries)
                timeseries = timeseries - theta_mean
            joint_power_spectrum = series_to_wavelets(timeseries,
                                                      num_windows=num_windows,
                                                      min_window_width=min_window_width,
                                                      max_window_width=max_window_width)
            pose_power_spectrum[idx:idx+num_windows] = joint_power_spectrum
            idx+=num_windows
    return pose_power_spectrum


max_distance_static = np.sqrt(2)*19

def calculate_static_synchrony(A, B, frame_range='all'):
    trackA = dict(A)
    trackB = dict(B)
    framesA, framesB = trackA['frame_ids'], trackB['frame_ids']
    if frame_range != 'all':
        assert isinstance(frame_range, tuple), 'If not "all", please input frame_range as tuple. Eg: (in_frame, out_frame).'
        framelist = np.arange(frame_range[0], frame_range[1])
        frameLocsA, frameLocsB = np.where(np.isin(framesA, framelist))[0], np.where(np.isin(framesB, framelist))[0]
        for key, value in trackA.items():
            trackA[key] = value[frameLocsA]
        for key, value in trackB.items():
            trackB[key] = value[frameLocsB]
        framecount = len(framelist)
    else:
        framecount = len(framesA)
        assert len(trackA['frame_ids']) == len(
            trackB['frame_ids']), "Tracks are of different framecounts. Please set frame range as a tuple."
    trackA, trackB = trackA['quaternion'], trackB['quaternion']
    static_sync_timeseries = []
    for i in range(framecount):
        pose1 = [Quaternion(j) for j in trackA[i]]
        pose2 = [Quaternion(j) for j in trackB[i]]
        distance = np.sum([Quaternion.absolute_distance(pose1[k], pose2[k]) for k in range(1,20)])
        static_sync = -(2 * (distance / max_distance_static)) + 1
        static_sync_timeseries.append(static_sync)
    return np.array(static_sync_timeseries)


def static_synchrony_all_avg(pose):
    """
    Args:
        pose: PsyPose pose object

    Returns:
        static_timeseries: Timeseries of average static synchrony per frame. Nans if <2 tracks present per frame.

    """
    track_occurence = {}
    data = pose.pose_data
    for frame in range(pose.framecount):
        present_tracks = []
        for key, val in data.items():
            if frame in val['frame_ids']:
                present_tracks.append(key)
        track_occurence[frame] = present_tracks
    pose.track_occurence = track_occurence
    out_vec=[]
    for frame in range(pose.framecount):
        tracks = track_occurence[frame]
        if len(tracks) < 2:
            out_vec.append(np.nan)
        else:
            quat_arrays = []
            for track in tracks:
                track_data = data[track]
                quat_arrays.append(track_data['quaternion'][np.where(track_data['frame_ids']==frame)[0][0]])
            sync_arr = np.empty((len(quat_arrays), len(quat_arrays)))
            for i in range(len(quat_arrays)):
                for j in range(len(quat_arrays)):
                    a = [Quaternion(q) for q in quat_arrays[i]]
                    b = [Quaternion(q) for q in quat_arrays[j]]
                    distance = np.sum([Quaternion.absolute_distance(a[k], b[k]) for k in range(1,20)])
                    static_sync = -(2 * (distance / max_distance_static)) + 1
                    sync_arr[i][j] = static_sync
            val = np.mean(sync_arr[np.tril_indices_from(sync_arr, k=-1)])
            #val = -(2*(val/max_distance)) + 1
            out_vec.append(val)
    return np.array(out_vec)

def calculate_meta_synchrony(A, B, frame_range='all', **kwargs):
    trackA = dict(A)
    trackB = dict(B)
    framesA, framesB = trackA['frame_ids'], trackB['frame_ids']
    if frame_range != 'all':
        assert isinstance(frame_range, tuple), 'If not "all", please input frame_range as tuple. Eg: (in_frame, out_frame).'
        framelist = np.arange(frame_range[0], frame_range[1])
        frameLocsA, frameLocsB = np.where(np.isin(framesA, framelist))[0], np.where(np.isin(framesB, framelist))[0]
        for key, value in trackA.items():
            trackA[key] = value[frameLocsA]
        for key, value in trackB.items():
            trackB[key] = value[frameLocsB]
        framecount = len(framelist)
    else:
        framecount = len(framesA)
    if framecount < 150:
        warnings.warn("If framecount is too small, meta-synchrony will likely be innacurate and reflect a parabolic shape.")
    assert len(trackA['frame_ids']) == len(trackB['frame_ids']), "Tracks are of different framecounts. Please set frame range as a tuple."
    trackA, trackB = trackA['quaternion'], trackB['quaternion']
    wavematA, wavematB = pose_to_wavelet_matrix(trackA, **kwargs), pose_to_wavelet_matrix(trackB, **kwargs)
    meta = np.corrcoef(wavematA.T, wavematB.T)[:framecount, :framecount][0]
    return meta

def calculate_meta_synchrony_average_all(pose, frame_range='all', exclude_under=150, **kwargs):
    track_occurence = {}
    data = pose.pose_data
    exclusion_tracks = []
    for track in data.keys():
        if len(data[track]['frame_ids'])<exclude_under:
            exclusion_tracks.append(track)
    for frame in range(pose.framecount):
        present_tracks = []
        for key, val in data.items():
            if key in exclusion_tracks:
                None
            elif frame in val['frame_ids']:
                present_tracks.append(key)
        track_occurence[frame] = present_tracks
    pose.track_occurence = track_occurence
    out_vec = []

    power_mats = {}
    for track, pdata in data.items():
        wavemat = pose_to_wavelet_matrix(pdata['quaternion'], **kwargs)
        frame_wavelet = dict(zip(['frame_ids', 'wavemat'], [pdata['frame_ids'], wavemat]))
        power_mats.update({track:frame_wavelet})

    meta_timeseries = []
    for frame in range(pose.framecount):
        tracks = track_occurence[frame]
        if len(tracks) < 2:
            out_vec.append(np.nan)
        else:
            power_spectrums = []
            for track in tracks:
                track_dat = power_mats[track]
                power_spectrums.append(track_dat['wavemat'][:, np.where(track_dat['frame_ids'] == frame)[0][0]])
            meta_timeseries.append(np.mean(np.corrcoef(power_spectrums)[0]))
    return meta_timeseries




# class Synchrony(object, pose):
#     def __init__(self):
#         self.roi = 'body'
#         self.pose = pose
#
#         pass
#
#     def set_roi(self, roi):
#         self.roi = roi
#
#
#
#
#
#







