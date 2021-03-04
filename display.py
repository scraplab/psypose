#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:44:01 2020

@author: f004swn
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import cv2
import networkx as nx
#import matplotlib
import glob
import scipy.stats
import random
import psypose.utils 
from keras.preprocessing.image import img_to_array, load_img
import shutil
from psypose import utils

#matplotlib.get_backend()

#os.chdir('/Users/f004swn/Documents')
#pickle = 'vibe_output.pkl'
#video = '/Users/f004swn/Documents/Code/packages/VIBE/sample_video.mp4'

#pickle = '/Users/f004swn/Documents/Code/pose_data/500_cut_unsquish.pkl'
#video = '/Users/f004swn/Documents/Code/pose_data/500_cut_unsquish.mp4'

#creating edges from list of tuples
# I now realize I can use get_spin_skeleton() for this (VIBE function)


spin_skel = [tuple(i) for i in [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [1 , 5],
            [5 , 6],
            [6 , 7],
            [1 , 8],
            [8 , 9],
            [9 ,10],
            [10,11],
            [8 ,12],
            [12,13],
            [13,14],
            [0 ,15],
            [0 ,16],
            [15,17],
            [16,18],
            [21,19],
            [19,20],
            [14,21],
            [11,24],
            [24,22],
            [22,23],
            [0 ,38],
        ]]
#oPickle = dict(joblib.load(pickle))
def plot_box(ax_obj, data, fb, color):
    if fb == 'body':
        cx, cy, w, h = [float(i) for i in data]
        top, right, bottom, left = [(cy-h/2), (cx+w/2), (cy+h/2), (cx-w/2)]
        ax_obj.vlines(x=[right, left], ymin=bottom, ymax=top, color=color)
        ax_obj.hlines(y=[top, bottom], xmin=left, xmax=right, color=color)
    elif fb == 'face':
        ax_obj.vlines(x=[data[1], data[3]], ymin=data[0], ymax=data[2], color=color)
        ax_obj.hlines(y=[data[0], data[2]], xmin=data[3], xmax=data[1], color=color)


def track(pose, trackID):
    video = pose.video_cv2
    pkl = pose.pose_data
    
    plt.ion()
    
    pkl = pkl[trackID]
    joints = pkl['joints3d']
    boxes = pkl['bboxes']
    track_frames = pkl['frame_ids']
    frameCount = len(track_frames)

    #here, the global flag makes the frameLoc variable available for the nested function to change
    global frameLoc
    frameLoc = int(track_frames[0])
    #frameLoc = np.where(np.asarray(ts)==min(ts, key=lambda x:abs(x-conv_ts)))[0][0]
    video.set(cv2.CAP_PROP_POS_FRAMES,frameLoc)
    ret, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    global pose_loc
    pose_loc = 0
    def switch_frame(event):
        global frameLoc
        global pose_loc
        if event.key == 'right':
            frameLoc+=1
            pose_loc+=1
        elif event.key == 'left':
            frameLoc-=1
            pose_loc-=1
        ax1.clear()
        video.set(cv2.CAP_PROP_POS_FRAMES,frameLoc)
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vid_frame = ax1.imshow(frame)
        ax1.axis('off')
        vid_frame.set_data(frame)
        plot_box(ax1, boxes[pose_loc], 'body', 'red')

        
        ts_graph = all_graphs[pose_loc]
        ax2.clear()
        ax2._axis3don = False
        ax2.set_xlim3d(-1,1)
        ax2.set_ylim3d(-1.3,0.7)
        ax2.set_zlim3d(-1,1)
        for key, value in graph_dicts[pose_loc].items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            ax2.scatter(xi, yi, zi, color='green')
            pos = nx.get_node_attributes(ts_graph, 'pos')
        for i, j in enumerate(ts_graph.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))   
            ax2.plot(x, y, z, color='blue')
                    
        fig.canvas.draw()

    fig = plt.figure(figsize=(20,10), dpi=100)
    fig.canvas.mpl_connect('key_press_event', switch_frame)
        
    ax1 = fig.add_subplot(121)
    ax1.axis('off')
    vid_frame = ax1.imshow(frame)
    plot_box(ax1, boxes[pose_loc], 'body', 'red')
    
    
    all_graphs = []
    graph_dicts = []
    for fr in range(frameCount):
        pkl_frame=None
        pkl_frame = joints[fr]
        netDict = {}
        for i in range(49):
            netDict.update({i:tuple(pkl_frame[i])})
        graph_dicts.append(netDict)
        sk=''
        sk = nx.Graph()
        sk.add_nodes_from(netDict.keys())
        for n, p in netDict.items():
            sk.nodes[n]['pos']=p
            sk.add_edges_from(spin_skel)
        all_graphs.append(sk)
    
    ts_graph = all_graphs[pose_loc]
    ax2 = fig.add_subplot(122, projection='3d')
    ax2._axis3don = False
    ax2.set_xlim3d(-1,1)
    ax2.set_ylim3d(-1.3,0.7)
    ax2.set_zlim3d(-1,1)
    for key, value in graph_dicts[pose_loc].items():
        xi = value[0]
        yi = value[1]
        zi = value[2]
        ax2.scatter(xi, yi, zi, color='green')
    pos = nx.get_node_attributes(ts_graph, 'pos')
    for i, j in enumerate(ts_graph.edges()):
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))   
        ax2.plot(x, y, z, color='blue')
    ax2.view_init(270,270)
    fig.suptitle(str(trackID), fontsize=50)

#plot = show_pose(video, pickle, 6)

def show_frame():
    #coming soon

    print('no')
    
    
tt
#display_pkl(video, pickle, '00:00:00', '/Users/f004swn/Documents/')

#pickle = '/Users/f004swn/Documents/Code/pose_data/500_cut_unsquish.pkl'
#video = '/Users/f004swn/Documents/Code/pose_data/500_cut_unsquish.mp4'
def pkl_to_array(pickle_path, video_path):
    data = dict(joblib.load(pickle_path))
    tracks = len(data)
    video = cv2.VideoCapture(video_path)
    frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    arr = np.zeros((tracks, frameCount))
    track_ids = list(data.keys())
    for t, track in enumerate(track_ids):
        frameIDs = data.get(track).get('frame_ids')
        arr[t][frameIDs]=1
    return arr
        
#pkl_arr = pkl_to_array(pickle, video)
def collapse(pickle_array):
    frames = pickle_array.shape[1]
    regr = np.zeros((frames))
    for frame in range(frames):
        regr[frame] = sum(pickle_array[:,frame])
    return regr

#pkl_regr = collapse(pkl_arr)


def show_tracks(pickle_array):
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=200, figsize=(10, 5), sharex=True)

    ax1.imshow(pickle_array, aspect='auto', interpolation='none', cmap='copper')
    for track in range(pickle_array.shape[0]):    
        ax1.hlines(track-0.5, xmin=0, xmax=pickle_array.shape[1], color='white')
    ax1.set(ylabel='Track ID')
    ax1.set(title='Person presence per track')
        
    ax2.plot(collapse(pickle_array))
    ax2.set(ylabel='# People', xlabel='Frame ID', title='Number of people on screen')
#show_tracks(pkl_arr)
#video_cap = cv2.VideoCapture(video)


def tr_resample(arr, video, tr, method='mode'):
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    fptr = int(frame_rate*tr // 1) #frames per TR (as integer)
    epochs = [arr[i:i + fptr] for i in range(0, len(arr), fptr)]
    out = []
    if method=='mode':
        func = lambda x: scipy.stats.mode(x)[0]
    elif method=='average':
        func = np.mean
    elif method=='min':
        func = np.min 
    elif method=='max':
        func = np.max 
    elif method=='median':
        func = np.median
        
    for ep in epochs:
        out.append(float(func(ep)))
    return np.asarray(out)

def cluster(pose, cluster_num):
    if type(cluster_num) == str:
        clusters = pose.named_clusters
    else:
        clusters = pose.clusters
    track_ids = clusters.get(cluster_num)
    face_data = pose.face_data.copy()
    cluster_face_data = face_data[face_data['track_id'].isin(track_ids)]
    images = []
    for r in range(len(cluster_face_data)):
        row = cluster_face_data.iloc[r]
        frame = row['frame_ids']
        top, right, bottom, left = [int(pos) for pos in row['locations']]
        image = psypose.utils.frame2array(frame, pose.video_cv2)[top:bottom, left:right]
        image = psypose.utils.resize_image(image, (100,100))
        images.append(image)
    if len(images) > 16:
        selection = np.arange(0, len(images), len(images)//16)
        images = [images[i] for i in selection][:16]
    fig = plt.figure(figsize=(10,10))
    for i, img in enumerate(images):
        sub = fig.add_subplot(4, 4, i+1)
        sub.axis('off')
        sub.imshow(img, interpolation='nearest')
    fig.suptitle(str(cluster_num), fontsize=50)

    



