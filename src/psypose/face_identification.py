#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:30:19 2021

@author: f004swn
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from psypose import utils
# from psypose.models import facenet_keras, deepface
from scipy.spatial.distance import pdist, squareform, cdist, euclidean
from sklearn.cluster import AgglomerativeClustering, KMeans
#
# def add_face_id(pose, overwrite=False, encoder='facenet', use_TR=False, out=None):
#
#     face_df = pd.DataFrame(pose.face_data)
#     # Remove possible nan rows
#     face_df = face_df.dropna(axis=0)
#     #removing negative values
#     face_df = face_df[face_df['FaceRectY']>=0]
#     face_df = face_df[face_df['FaceRectX']>=0]
#     faces_to_process = int(face_df.shape[0])
#     unique_frames = [int(i) for i in np.unique(face_df['frame'])]
#
#     #if encoder=='default':
#     #    encoding_length = 128
#     #    encode = utils.default_encoding
#     if encoder=='facenet':
#         encoding_length = 128
#         encode = facenet_keras.encode
#     elif encoder=='deepface':
#         encoding_length = 2622
#         encode = deepface.encode
#
#     encoding_array = np.empty((faces_to_process, encoding_length))
#
#     print("Encoding face identities...\n", flush=True)
#     pbar = tqdm(total=faces_to_process)
#     counter = -1
#     for frame in unique_frames:
#         img = utils.frame2array(frame, pose.vid_cv2)
#         sub = face_df[face_df['frame']==frame]
#         for loc in range(sub.shape[0]):
#             row = sub.iloc[loc]
#             bbox = row[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']]
#             face_cropped = utils.crop_face(img, bbox)
#             encoding = encode(face_cropped)
#             counter+=1
#             encoding_array[counter] = encoding
#             pbar.update(1)
#     pbar.close()
#
#     encoding_columns = ['enc'+str(i) for i in range(encoding_length)]
#     face_df[encoding_columns] = encoding_array
#
#     face_df = face_df.reset_index(drop=True)
#
#     pose.face_data = face_df
#     pose.faces_encoded = True


def cluster_ID(pose, metric='cosine', linkage='average', overwrite=False, use_cooccurence=True):
    """
    Clusters all tracks based on facial encodings attached to each track.
    It is recommended to use the cosine metric and average linkage for best results.
    Outputs a dictionary where keys are cluster ID's and values are lists of tracks.

    """

    if overwrite:
        pose.clusters = None

    if pose.is_clustered and not overwrite:
        raise Exception('Pose object has already been clustered. Set overwrite=True to overwrite previous clustering.')

    face_data = pose.face_data
    pose_data = pose.pose_data

    pose.encoding_length = len([i for i in list(face_data.columns) if 'enc' in i])


    # Here, iterating through all rows of the face_rec data frame (all available frames)
    # checking if each frame is listed within any of the PARE tracks
    # if overlap frames are detected, the algo will check if the face box is within
    # the PARE bounding box. If True, then that frame will get that track ID

    fr_ids = []

    for r in range(len(face_data)): # to-do: update the method with which you iterate the df with the dict method (faster)
        row = face_data.iloc[r]
        frame = int(row['frame'])
        track_id_list = []
        for t in np.unique(list(pose_data.keys())):
            track = pose_data[t]
            track_frames = track['frame_ids']
            if frame in track_frames:
                track_id_list.append(t)
        if len(track_id_list) != 0:
            for track_id in track_id_list:
                box_loc = np.where(pose_data[track_id]['frame_ids'] == frame)[0][0]
                box = pose_data[track_id]['bboxes'][box_loc]
                if utils.check_match(box, utils.get_bbox(row)):
                    track_designation = int(track_id)
                    break
                else:
                    # track designation is 'no_match' if the face is not within body box,
                    # those rows will be removed.
                    track_designation = 'no_match'
            fr_ids.append(track_designation)
        else:
            # if there aren't any present track ID's in the same frame as the face was detected, no match
            fr_ids.append('no_match')
    face_data['track_id'] = fr_ids # add column with track ID designation to face data df
    pose.face_data = face_data

    # removing face encodings with no match
    face_data = face_data[face_data['track_id'] != 'no_match']

    # here, I'm going through each unique track and getting an average face encoding
    enc_columns = [i for i in face_data.columns if 'enc' in i]
    avg_encodings = []
    enc_tracks = []
    for loc, track in enumerate([int(i) for i in np.unique(face_data['track_id'])]):
        tr_df = face_data[face_data['track_id'] == track]
        avg_encodings.append(np.mean(tr_df[enc_columns].to_numpy(), axis=0))
        enc_tracks.append(track)

    track_enc_avgs = np.array(avg_encodings)
    track_encoding_avgs = dict(zip(enc_tracks, avg_encodings))

    # Here, we cluster all of the averaged track encodings.
    # Tracks containing the same face will be concatenated later

    # Here, I'm going to compute the between-track distances using only co-occuring tracks
    # I'll first create a co-occurence matrix.
    n_tracks = len(np.unique(face_data['track_id']))

    # Getting a list of track ID's with faces in the
    opt_tracks = np.unique(face_data['track_id']).tolist()
    # Initialize empty co-occurence matrix
    cooc = np.zeros((n_tracks, n_tracks))
    for i, track1 in enumerate(opt_tracks):
        frames_1 = pose_data[track1]['frame_ids']
        for j, track2 in enumerate(opt_tracks):
            frames_2 = pose_data[track2]['frame_ids']
            if len(np.intersect1d(frames_1, frames_2)) > 0: # this is asking if the two frame lists have at least one common frame
                cooc[i][j] = True
    cooc = cooc[np.tril_indices_from(cooc, k=-1)]

    # get per-track average encodings with nested list comprehension:
    # for every track with a face match, get every encoding and average it, put it into a list
    track_encoding_avgs = [np.mean(enc, axis=0) for enc in [face_data[face_data['track_id'] == track][enc_columns].to_numpy() for track in opt_tracks]]

    # store the average face encodings with track IDs as keys
    encoding_averages = dict(zip(opt_tracks, track_encoding_avgs))
    pose.track_encoding_avgs = encoding_averages # To-do: save this to .pose file
    avg_enc_array = np.empty((len(encoding_averages), pose.encoding_length)) # create avg encoding array for clustering
    for i, track in enumerate(list(encoding_averages.keys())):
        avg_enc_array[i] = encoding_averages[track]
    pose.average_encoding_array = avg_enc_array

    #make distance matrix using the defined metric
    encoding_dist_mat = squareform(pdist(track_encoding_avgs, metric=metric))
    #get upper triangle
    all_track_dist = encoding_dist_mat[np.tril_indices_from(encoding_dist_mat, k=-1)]
    inter_track_distances = np.extract(cooc, all_track_dist)
    # Get the within-track distances:
    # For each track, a distance matrix of all encodings is made. Then, the average of the upper triangle is made.
    intra_track_encodings = [face_data[face_data['track_id'] == tr][enc_columns] for tr in opt_tracks]
    # averaging distances for each track
    intra_track_distances = np.array([np.mean(pdist(encs, metric=metric)) for encs in intra_track_encodings]) # pdist returns the condensed (vectorized upper triangle) matrix
    intra_track_distances = intra_track_distances[~np.isnan(intra_track_distances)] # remove nans

    # To get a cutoff metric for agglomerative clustering, we cluster the inter-track and intra-track distances.
    # This ideally will create a bimodal distribution of distances, giving the clustering algo an idea
    # of how similar same-person tracks are and how different different-person tracks are.
    all_distances = np.concatenate((inter_track_distances, intra_track_distances))
    all_distances = all_distances.reshape(-1, 1)
    # Cluster the set of intra- and inter-track encoding distances into two
    kmeans = KMeans(n_clusters=2).fit(all_distances)
    # Get the distances present in each cluster.
    clus0, clus1 = all_distances[np.where(kmeans.labels_ == 0)], all_distances[np.where(kmeans.labels_ == 1)]
    # Get the distance value that is directly in the middle of the two clusters/distance distributions.
    # This becomes the distance_threhold parameter used in AgglomerativeClustering
    cut = float(max(clus0) + (min(clus1) - max(clus0)) / 2)
    # This final clustering uses the cut as a distance threshold.
    # Tracks in the same cluster will now be consolidated.
    final_clust = AgglomerativeClustering(linkage=linkage, distance_threshold=cut,
                                          n_clusters=None, affinity=metric).fit(track_enc_avgs)

    pose.face_clustering = final_clust

    # Now reconciling the actual track IDs that correspond to each cluster, and saving cluster IDs as key
    # in a dictionary sorted by the the number of tracks in each cluster. The values in the dict are track IDs.

    clusters = []
    cluster_tracks = []
    enc_tracks = np.array(enc_tracks)
    for clus in np.unique(final_clust.labels_):
        clusters.append(clus)
        tracks = enc_tracks[np.where(final_clust.labels_ == clus)[0]]
        cluster_tracks.append(tracks)
    fc = dict(zip(clusters, cluster_tracks))
    cluster_mains = [k for k in sorted(fc, key=lambda k: len(fc[k]), reverse=True)]
    t = [fc.get(clust) for clust in cluster_mains]
    sorted_dict = dict(zip([i for i in range(0, len(t))], t))

    # add cluster ids to face_data
    pose.clusters = sorted_dict
    pose.is_clustered = True

def name_clusters(pose, character_dict, overwrite_names=False):
    # This function takes a dictionary identifying clusters, such that character names are keys and
    # lists of cluster IDs are values.
    pose.character_key = character_dict
    # Get list of characters
    chars = list(np.unique(character_dict.keys())[0])
    # Iterate through the character key by name, make lists of all track IDs matched to that character from cluster
    names_clustered = {}
    for char in chars:
        # common_clusters = [k for k, v in character_dict.items() if str(v) == char]
        tracks = []
        for cluster in character_dict[char]:
            cluster_tracks = pose.clusters[cluster]
            # concatenate all lists of track ID's associated with each cluster into one
            tracks.extend(cluster_tracks)
        names_clustered.update({char: tracks})
    #names_sorted = sorted(names_clustered, key=lambda k: len(names_clustered[k]), reverse=True)
    #tracks_sorted = [names_clustered[name] for name in names_sorted]
    #names_clustered = dict(zip(names_sorted, tracks_sorted))
    pose.named_clusters = names_clustered
    pose.clusters_named = True
