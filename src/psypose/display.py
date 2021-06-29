#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 14:44:01 2020

@author: f004swn
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import cv2
import networkx as nx
#import matplotlib
import glob
import scipy.stats
import random
from keras.preprocessing.image import img_to_array, load_img
import shutil
from psypose import utils
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import base64
from io import BytesIO
from IPython.display import HTML

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

    #change this to utils.frame2array()?

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

def frame(pose, frame_number):
    frame = utils.frame2array(frame_number, pose.video_cv2)
    f = px.imshow(frame)
    f.show()
    del frame
    
def render_track(pose, track, format='mp4', outdir=None, loop=None):
    if not outdir and format=='mp4':
        outdir=str(track)+'.mp4'
    elif not outdir and format=='gif':
        outdir=str(track)+'.gif'
    vid = pose.video_cv2
    #frameCount = pose.n_frames
    fps = pose.fps
    
    joints = pose.pose_data[track]['joints3d']
    frame_ids =  pose.pose_data[track]['frame_ids']
    bboxes = pose.pose_data[track]['bboxes']
    
    matplotlib.use('Agg')
    
    img_array = np.empty((len(frame_ids), 720, 1280, 3), dtype='uint8')
    px = px = 1/plt.rcParams['figure.dpi']

    print("Processing video..\n")
    
    for fr, cur_frame in tqdm(enumerate(frame_ids)):
        frame = utils.frame2array(cur_frame, vid)
        pkl_frame = joints[fr]
        fig = plt.figure(figsize=(1280*px, 720*px))
        
        ax1 = fig.add_subplot(121)
        ax1.axis('off')
        ax1.imshow(utils.crop_image_body(frame, bboxes[fr]))
        #plot_box(ax1, bboxes[fr], 'body', 'red')
        
        netDict = {}
        for i in range(49):
            netDict.update({i:tuple(pkl_frame[i])})
        sk=None
        sk = nx.Graph()
        sk.add_nodes_from(netDict.keys())
        for n, p in netDict.items():
            sk.nodes[n]['pos']=p
        sk.add_edges_from(spin_skel)
        
        ax2 = fig.add_subplot(122, projection='3d')
        ax2._axis3don = False
        ax2.set_xlim3d(-0.5,0.5)
        ax2.set_ylim3d(-0.75,0.5)
        ax2.set_zlim3d(-0.5,0.55)
        for key, value in netDict.items():
            xi = value[0]
            yi = value[1]
            zi = value[2]
            ax2.scatter(xi, yi, zi, color='green')
        pos = nx.get_node_attributes(sk, 'pos')
        for i, j in enumerate(sk.edges()):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))   
            ax2.plot(x, y, z, color='blue')
        ax2.view_init(270, 270)
        fig.canvas.draw()
        out_frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        ##############################
        out_frame = out_frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_array[fr,:,:,:] = out_frame
        plt.close()
    if format=='mp4':
        utils.write2vid(img_array, fps, outdir, (1280, 720))
    elif format=='gif':
        print("Writing to gif...", flush=True)
        imgs = [Image.fromarray(img) for img in img_array]
        # duration is the number of milliseconds between frames; this is 40 frames per second
        dur = int(round((1/pose.fps)*1000))
        if loop == None:
            loop = 0
        imgs[0].save(outdir, save_all=True, append_images=imgs[1:], duration=dur, loop=loop)
#display_pkl(video, pickle, '00:00:00', '/Users/f004swn/Documents/')

#pickle = '/Users/f004swn/Documents/Code/pose_data/500_cut_unsquish.pkl'
#video = '/Users/f004swn/Documents/Code/pose_data/500_cut_unsquish.mp4'
def pkl_to_array(pose):
    data = pose.pose_data
    tracks = len(data)
    #video = cv2.VideoCapture(video_path)
    #frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    arr = np.zeros((tracks, pose.framecount))
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


def show_tracks(pose, dpi=100, figsize=(10,5)):
    pkl_arr = pkl_to_array(pose)
    #pkl_arr = collapse(pkl_arr)

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, dpi=dpi, figsize=figsize, sharex=True)

    ax1.imshow(pkl_arr, aspect='auto', interpolation='none', cmap='copper')
    for track in range(pkl_arr.shape[0]):    
        ax1.hlines(track-0.5, xmin=0, xmax=pkl_arr.shape[1], color='white')
    ax1.set(ylabel='Track ID')
    ax1.set(title='Track Occurrence')
        
    ax2.plot(collapse(pkl_arr))
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
    enc_columns = [i for i in pose.face_data.columns if 'enc' in i]
    if isinstance(cluster_num, str):
        clusters = pose.named_clusters
    else:
        clusters = pose.clusters
    track_ids = clusters.get(cluster_num)
    face_data = pose.face_data.copy()
    cluster_face_data = face_data[face_data['track_id'].isin(track_ids)]
    images = []
    for r in range(len(cluster_face_data)):
        row = cluster_face_data.iloc[r]
        frame = row['frame']
        cx, cy, w, h = utils.get_bbox(row)
        top, right, bottom, left = [int(round(i)) for i in [cy, cx+w, cy+h, cx]]
        image = utils.frame2array(frame, pose.video_cv2)[top:bottom, left:right]
        image = utils.resize_image(image, (100,100))
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
    
def face(pose, face_loc):
    info = pose.face_data.iloc[face_loc]
    frame = info['frame']
    bbox = utils.get_bbox(info)
    image = utils.frame2array(frame, pose.video_cv2)
    image = utils.crop_face(image, bbox)
    plt.imshow(image)

def play_video(pose):
    # meant to play video in colab 
    mp4 = open(pose.vid_path,'rb').read()
    data_url = "data:video/mp4;base64," + base64.b64encode(mp4).decode()
    HTML("""
    <video width=400 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)


######## Begin code for 3d viewer #########

def get_frame_info(in_data, idx):
    # in_data containes the complete track data,
    # pose data, video array, bboxes, frame ids
    one_pose = in_data['pose_data'][idx]
    one_bbox = in_data['bboxes'][idx]
    one_frame_id = in_data['frame_ids'][idx]
    one_frame_image = utils.frame2array(one_frame_id, in_data['vid'])
    #one_frame_image = in_data['vid_bytes'][idx]
    frame_data = {
        'pose':one_pose,
        'bbox':one_bbox,
        'image':one_frame_image,
        'frame_id':one_frame_id
    }
    return frame_data

def scatter_fig(fig, x_dat, y_dat, z_dat):
  # this generates a single frame of the 3d skeleton. 
    x, y, z = [i.flatten() for i in [x_dat, y_dat, z_dat]]
    netDict = {}
    for i in range(49):
        netDict.update({i:tuple([x[i], y[i], z[i]])})
    sk = nx.Graph()
    sk.add_nodes_from(netDict.keys())
    for n, p in netDict.items():
        sk.nodes[n]['pos']=p
        sk.add_edges_from(spin_skel)
    Edges = list(sk.edges())
    Nodes = list(sk.nodes())
    node_attrs = nx.get_node_attributes(sk, 'pos')
    N = sk.number_of_nodes()
    Xn=[node_attrs[i][0] for i in range(N)]
    Yn=[node_attrs[i][1] for i in range(N)]# y-coordinates
    Zn=[node_attrs[i][2] for i in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in list(sk.edges()):
        Xe+=[node_attrs[e[0]][0],node_attrs[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[node_attrs[e[0]][1],node_attrs[e[1]][1], None]
        Ze+=[node_attrs[e[0]][2],node_attrs[e[1]][2], None]

    axis=dict(
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis)))
    trace1=go.Scatter3d(x=Xe,
                  y=Ye,
                  z=Ze,
                  mode='lines',
                  line=dict(color='black', width=4),
                  hoverinfo='none',
                  )

    trace2=go.Scatter3d(x=Xn,
                  y=Yn,
                  z=Zn,
                  mode='markers',
                  marker=dict(symbol='circle',
                                size=6,
                                line=dict(color='blue', width=0.075)
                                ),
                    hoverinfo='none')

    fig.add_trace(trace1, row=1, col=2)
    fig.add_trace(trace2, row=1, col=2)

def scatter_fig_singleframe(fig, x_dat, y_dat, z_dat):
  # this generates a single frame of the 3d skeleton. 
    x, y, z = [i.flatten() for i in [x_dat, y_dat, z_dat]]
    netDict = {}
    for i in range(49):
        netDict.update({i:tuple([x[i], y[i], z[i]])})
    sk = nx.Graph()
    sk.add_nodes_from(netDict.keys())
    for n, p in netDict.items():
        sk.nodes[n]['pos']=p
        sk.add_edges_from(spin_skel)
    Edges = list(sk.edges())
    Nodes = list(sk.nodes())
    node_attrs = nx.get_node_attributes(sk, 'pos')
    N = sk.number_of_nodes()
    Xn=[node_attrs[i][0] for i in range(N)]
    Yn=[node_attrs[i][1] for i in range(N)]# y-coordinates
    Zn=[node_attrs[i][2] for i in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in list(sk.edges()):
        Xe+=[node_attrs[e[0]][0],node_attrs[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[node_attrs[e[0]][1],node_attrs[e[1]][1], None]
        Ze+=[node_attrs[e[0]][2],node_attrs[e[1]][2], None]

    axis=dict(
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          title=''
          )

    layout = go.Layout(
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis)))
    trace1=go.Scatter3d(x=Xe,
                  y=Ye,
                  z=Ze,
                  mode='lines',
                  line=dict(color='black', width=4),
                  hoverinfo='none',
                  )

    trace2=go.Scatter3d(x=Xn,
                  y=Yn,
                  z=Zn,
                  mode='markers',
                  marker=dict(symbol='circle',
                                size=6,
                                line=dict(color='blue', width=0.075)
                                ),
                    hoverinfo='none')

    fig.add_trace(trace1)
    fig.add_trace(trace2)

def body3d(fig, pose_data):
    # in x -> z -> y order because the format of the output is not intuitive
    x, z, y = pose_data[:,0]*-1, pose_data[:,1]*-1, pose_data[:,2]*-1
    scatter_fig(fig, x, y, z)
    fig.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, range=[-0.75, 0.75], showbackground=False),
        yaxis=dict(showticklabels=False, range=[-0.75, 0.75], showbackground=False),
        zaxis=dict(showticklabels=False, range=[-1.0, 1.25], showbackground=True),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=2)
        ), showlegend=False)

def body3d_singleframe(fig, pose_data):
    # in x -> z -> y order because the format of the output is not intuitive
    x, z, y = pose_data[:,0]*-1, pose_data[:,1]*-1, pose_data[:,2]*-1
    scatter_fig_singleframe(fig, x, y, z)
    fig.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, range=[-0.75, 0.75], showbackground=False),
        yaxis=dict(showticklabels=False, range=[-0.75, 0.75], showbackground=False),
        zaxis=dict(showticklabels=False, range=[-1.0, 1.25], showbackground=True),
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=2)
        ), showlegend=False)

def frame3d(pose, track, idx):
    fig = go.Figure()
    pose_data = pose.pose_data[track]['joints3d'][idx]
    body3d_singleframe(fig, pose_data)
    fig.show()

#def extract_body_image(array, data):
    # This takes an image in numpy format and a body bbox, crops the image, and scales it down to 100x100
#    abs_h, abs_w = array.shape[0], array.shape[1]
#    cx, cy, w, h = [i for i in data]
#    top, right, bottom, left = [int(round(i)) for i in [(cy-h/2), int(cx+w/2), int(cy+h/2), (cx-w/2)]]#

    # fixing the issue when bbox is partially out of frame. 
    # Later I wish to change this so that images are padded with black rather than 
    # making the edges=0 when it is partially out of frame. 
#    if right > abs_w:
#      right = abs_w
#    if left < 0:
#        left = 0
#    if bottom > abs_h:
#      bottom = abs_h
#    if top < 0:
#        top = 0

#    new_img = array[top:bottom, left:right, :]
#    out_img = utils.resize_image(new_img, (100,100))
#    return out_img

def extract_body_image(array, data):
    # This takes an image in numpy format and a body bbox, crops the image, and scales it down to 100x100
    abs_h, abs_w = array.shape[0], array.shape[1]
    cx, cy, w, h = [i for i in data]
    top, right, bottom, left = [int(round(i)) for i in [(cy-h/2), int(cx+w/2), int(cy+h/2), (cx-w/2)]]

    # Padding images with black if the bbox is out-of-frame

    if right > abs_w:
      r_overhang = right-round(abs_w) + 10
      array = np.pad(array, ((0,r_overhang),(0,0),(0,0)))
    if left < 0:
      l_overhang = -1*left
      right-=left
      left = 0
      array = np.pad(array, ((l_overhang,0),(0,0),(0,0)))    
    if bottom > abs_h:
      b_overhang = bottom-round(abs_h) + 10
      array = np.pad(array, ((0,0),(0,b_overhang),(0,0)))
    if top < 0:
      t_overhang = -1*top
      bottom-=top
      top=0
      array = np.pad(array, ((0,0),(t_overhang,0),(0,0)))

    new_img = array[top:bottom, left:right, :]
    out_img = utils.resize_image(new_img, (100,100))
    return out_img

def draw_box(fig, bbox, color='Red'):
    # draws the body bbox on to the left subplot (the video frame)
    cx, cy, w, h = [float(i) for i in bbox]
    top, right, bottom, left = [(cy-h/2), (cx+w/2), (cy+h/2), (cx-w/2)]
    fig.add_shape(type="rect", xref="x", yref="y", x0=left, y0=top, x1=right, y1=bottom,
        line=dict(
            color=color,
            width=2), row=1, col=1)

def img_to_b64(arr_img):
    pil_img = Image.fromarray(arr_img)
    prefix = "data:image/png;base64,"
    with BytesIO() as stream:
        pil_img.save(stream, format="png")
        base64_string = prefix + base64.b64encode(stream.getvalue()).decode("utf-8")
    return base64_string

def add_bbox_frame(fig, image_str, bbox):
    fig.add_trace(go.Image(source=image_str), row=1, col=1)
    fig.update_xaxes(range=[0,100], constrain='domain', scaleanchor='y', scaleratio=1, row=1, col=1)
    fig.update_yaxes(range=[100,0], constrain='domain', scaleanchor='x', scaleratio=1, row=1, col=1)
    #draw_box(fig, bbox=bbox)

def pose_subplot(in_data, idx, plot_type):
    frame_data = get_frame_info(in_data, idx)
    vid_shape = in_data['shape'] # (height, width)
    frame_id = frame_data['frame_id']
    vid_image = frame_data['image']
    bbox = frame_data['bbox']
    extr_body_image = extract_body_image(vid_image, bbox)
    img_str = img_to_b64(extr_body_image)
    #img_str = frame_data['b64_img']
    pose = frame_data['pose']
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "scene"}]],
                        subplot_titles=('Target Body', 'Pose Estimate'))
    # Add video frame + bbox trace with the show_bbox_frame function
    add_bbox_frame(fig, img_str, bbox)
    # Add 3d pose figure
    body3d(fig, pose)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    if plot_type=='init':
      return fig
    elif plot_type=='frame':
      out_dict = fig.to_dict()
      out_dict['name'] = str('f'+str(frame_id))
      return go.Frame(out_dict)


def make_step(frame_id, dur):
    out_step = {
    'method': 'animate',
    'label': str(frame_id),
    'args': [['f'+str(frame_id)], {'frame': {'duration': dur, 'redraw': True}, 'mode': 'immediate'}]
    }
    return out_step
    

def slider_dict(steps):
    sliders_dict = {
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 20},
            'prefix': 'Frame:',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 0},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': steps,

    }
    return sliders_dict

def play_pause(dur):
    buttons = {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': dur, 'redraw': True},
                         'fromcurrent': True, 'transition': {'duration': 0}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration': dur, 'redraw': True}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
    return [buttons]

def track3d(pose, track_id, export_to_path=None):
    print("Generating plot...\n", flush=True)
    dur = str(int(round((1/pose.fps)*1000)))
    data = pose.pose_data[track_id]
    #vid_array = pose.video_array
    vid = pose.video_cv2
    frame_ids = data['frame_ids']
    pose_data = data['joints3d']
    bboxes = data['bboxes']
    n_frames = len(frame_ids)
    vid_shape = pose.video_shape
    #in_data = {'vid':vid_array, 'shape':vid_shape,'frame_ids':frame_ids, 'pose_data':pose_data, 'bboxes':bboxes, 'n_frames':n_frames}
    in_data = {'vid':vid, 'shape':vid_shape,'frame_ids':frame_ids, 'pose_data':pose_data, 'bboxes':bboxes, 'n_frames':n_frames}
    fig = pose_subplot(in_data, 0, 'init')
    frames = [pose_subplot(in_data, i, 'frame') for i in tqdm(range(n_frames), position=0, leave=True)] 
    # testing a new pbar implementation
    #plot_progress = [process(token) for token in tqdm(frames)]
    steps = [make_step(i, dur) for i in [frame_ids[j] for j in range(n_frames)]]
    #sliders_dict = slider_dict(steps)
    fig.update(frames=frames)
    #fig.layout['sliders']=[sliders_dict]
    fig.layout['updatemenus'] = play_pause(dur)
    fig.update_layout(height=600, width=1000, title_text="Track "+str(track_id))
    fig.show()
    if export_to_path != None:
        fig.write_html(export_to_path)

############# Begin code for grid view of faces #################


def extract_face_image(array, data):
    # This takes an image in numpy format and a face bbox, crops the image, and scales it down to 50x50
    #abs_h, abs_w = array.shape[0], array.shape[1]
    cx, cy, w, h = [i for i in data]
    top, right, bottom, left = [int(round(i)) for i in [cy, cx+w, cy+h, cx]]
    new_img = array[top:bottom, left:right, :]
    out_img = utils.resize_image(new_img, (50,50))
    return out_img

def retrieve_face(pose, trackID):
    df = pose.face_data
    df = df[df['track_id']==trackID]
    loc = df.loc[df['FaceScore'].idxmax()]
    bbox, frame = utils.get_bbox(loc), int(loc['frame'])
    in_image = utils.frame2array(frame, pose.video_cv2)
    face_arr = extract_face_image(in_image, bbox)
    out = img_to_b64(face_arr)
    return out


def add_layout_image(figure, b_string, x, y, sizex, sizey):
    figure.add_layout_image(dict(source=b_string,
                            xref='x',
                            yref='y',
                            xanchor='left',
                            yanchor='bottom',
                            x=x,
                            y=y,
                            sizing="contain",
                            sizex=sizex,
                            sizey=sizey,
                            layer="above"
                          )
                      )


def clusters(pose, type='unnamed'):
  df = pd.DataFrame(columns=['trackID', 'clusterID'])
  df['trackID'], df['clusterID'] = pose.track_encoding_avgs.keys(), pose.face_clustering.labels_
  f = px.scatter()
  #f.update_layout(yaxis=dict(autorange='reversed'))
  unq_tracks = np.unique(df['trackID'])
  unq_clusters = np.unique(df['clusterID'])
  markers_x, markers_y, trackIDs, clusterIDs = [], [], [], []
  for y, j in enumerate(unq_clusters):
      clus_df = df[df['clusterID']==j]
      tracks = np.unique(clus_df['trackID'])
      for x, l in enumerate(tracks):
          img_str = retrieve_face(pose, l)
          add_layout_image(f, img_str, x, y, 1, 1)
          markers_x.append(x+0.5)
          markers_y.append(y+0.5)
          trackIDs.append(l)
          clusterIDs.append(j)
  marker_frame = pd.DataFrame(dict(zip(['x', 'y', 'track', 'cluster'],
                                       [markers_x, markers_y, trackIDs, clusterIDs])))
  f.update_layout(height=1000, width=1000, xaxis=dict(range=[0, 15], showgrid=False, showticklabels=False, title='Tracks'), 
                  yaxis=dict(range=[0, 15], gridwidth=2, nticks=16, title='Cluster ID'))
  
  f.add_trace(go.Scatter(x=markers_x, y=markers_y, hovertext=['Track '+str(i) for i in trackIDs], hoverinfo='text', showlegend=False, mode="markers"))
  f.update_layout(hoverlabel=dict(bgcolor='white'))
  f.show(config={"modeBarButtonsToRemove":['zoom2d', 'toggleSpikeLines', 'lasso2d', 'autoscale2d', 'select2d'], 
                 "displayModeBar":True})


    



