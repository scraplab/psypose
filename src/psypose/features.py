#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Functions for the construction of second-order pose data and regressors

"""
import numpy as np
import pandas as pd
import scipy.stats
from psypose import utils
import os

def imaging_pars(pose, functional = 'a_func_file', TR=2.0):
    #make a function that allows you to either manually input imaging parameters
    #or provide a sample functional run for parameters (HRF, TR, etc)
    #this info will be used to generate the different regressors
    pose.TR = TR


def synchrony_matrix(pose):
    dat = pose.pose_data
    framecount = pose.framecount
    dk = list(dat.keys())  # list of tracks
    ntt = len(dk)  # number of tracks
    # make presence matrix
    ptmat = utils.make_presence_mat(pose)

    # compute synchrony
    cid = np.where(ptmat)[0]  # get those track indexes here ** np.where() will return indices where number is not 0.
    nj = 25  # number of joints
    dsel = np.tril_indices(nj, k=-1)  # indices of lower triangle of joint similarity matrix
    out = []

    # for every frame, make an array that represents the joint distance mat of every every individual (in a subloop)
    # after it's made for each frame, get the correlation matrix of distance matrices.  These represent the synchyrony matrix for all individuals
    dk = np.array(dk)
    # for i in tqdm(range(framecount)): # for every frame
    n_people = []
    for i in tqdm(range(framecount)):
        track_ids = dk[np.where(ptmat[i, :])[0]]
        n_presence = len(track_ids)
        n_people.append(n_presence)
        if not n_presence:
            out.append(np.nan)
        else:
            dmats = np.zeros((300, n_presence))  # ((25*25)-25)/2, aka len() of the lower triangle of the sync mat
            for j in range(
                    n_presence):  # for every track, get the distance matrix of that track with all other tracks, then subset to the lower triangle
                # track_ids = dk[np.where(ptmat[:,i])[0]]
                pose_loc = np.where(dat[track_ids[j]]['frame_ids'] == i)[0][0]
                pose = dat[track_ids[j]]['joints3d'][pose_loc, :25, :]
                dmats[:, j] = pairwise_distances(pose)[dsel]
            out.append(np.corrcoef(dmats, rowvar=False))
    return out

def average_synchrony(pose):

    #### the code belo
    dat = pose.pose_data
    framecount = pose.framecount
    out = synchrony_matrix(pose)

    # for every frame, make an array that represents the joint distance mat of every every individual (in a subloop)
    # after it's made for each frame, get the correlation matrix of distance matrices.  These represent the synchyrony matrix for all individuals
    dk = np.array(dk)
    # for i in tqdm(range(framecount)): # for every frame
    n_people = []
    for i in tqdm(range(framecount)):
        track_ids = dk[np.where(ptmat[i, :])[0]]
        n_presence = len(track_ids)
        n_people.append(n_presence)
        if not n_presence:
            out.append(np.nan)
        else:
            dmats = np.zeros((300, n_presence))  # ((25*25)-25)/2, aka len() of the lower triangle of the sync mat
            for j in range(
                    n_presence):  # for every track, get the distance matrix of that track with all other tracks, then subset to the lower triangle
                # track_ids = dk[np.where(ptmat[:,i])[0]]
                pose_loc = np.where(dat[track_ids[j]]['frame_ids'] == i)[0][0]
                pose = dat[track_ids[j]]['joints3d'][pose_loc, :25, :]
                dmats[:, j] = pairwise_distances(pose)[dsel]
            out.append(np.corrcoef(dmats, rowvar=False))  # fill this person-person synchrony matrix in the 'out' array

    # out now represesents a synchrony matrix for every frame

    # compute average synchrony

    # making a list of subset indices
    tsel = []
    for i in out:
        if isinstance(i, float):
            tsel.append(np.nan)
        else:
            tsel.append(np.tril_indices_from(i, k=-1))

    gas = np.zeros(framecount)  # empty vec that is n_frames long
    smin = np.zeros(framecount)  # empty vec that is n_frames long

    # for every frame,
    # get the synchrony matrix,
    # get the average of the subset matrix (lower triangle)
    # get the minimum of that subet matrix
    # fill the gas with the averages and smin with the minimums
    for i in range(framecount):
        kout = out[i]
        if isinstance(kout, float):
            gas[i], smin[i] = np.nan, np.nan
        else:
            gas[i], smin[i] = np.mean(kout[tsel[i]]), np.min(kout[tsel[i]])
    gas = (gas + 1) / 2
    return gas

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

    if type=='dynamic':
        max_distance = 2*static_max

    out_vec=[]
    for frame in range(pose.framecount):
        tracks = track_occurence[frame]
        if len(tracks) < 2:
            out_vec.append(np.nan)
        else:
            pose_vectors = []
            for track in tracks:
                track_data = data[track] # need to update dynamic
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
            val = -(2*(val/static_max)) + 1
            out_vec.append(val)
    return np.array(out_vec)


def series_to_wavelets(data, num_windows=10, min_window_width=10, max_window_width=90):
    """
    Convert any timeseries into wavelet matrix that is n_timepoints x num_windows
    @param data: Input timeseries.
    @type data: iterable
    @param num_windows: Number of frequency windows.
    @type num_windows: int
    @param min_window_width: The minimum period of timepoints to consider.
    @type min_window_width: int
    @param max_window_width: The maximum period of timepoints to consider.
    @type max_window_width: int
    @return: Power spectrum wavelet matrix.
    @rtype: ndarray
    """
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


def static_synchrony_all_avg(pose, frame_range='all'):
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

def preproccess_pose(pose, **kwargs):
    """
    Preprocess the pose data to fit desire parameters defined in **kwargs.
    @param pose: PsyPose pose object.
    @type pose: object
    @param kwargs:
    @type kwargs:
    @return:
    @rtype:
    """

class Synchrony(object):

    """
    Class for generating different synchrony measures time series.
    """

    def __init__(self):
        self.roi = 'body' # set to all
        self.pose = None
        self.joints = 'all' # upper, lower, all, or list
        self.ignore_root = True # ignore root joint (pelvis)?
        self.frame_range = 'all'
        self.specificity = 'all' # or call it average, and let it be all or something
        self.track_interest = None
        self.laterality = 'mirrored' # naming convention ideas: agnostic, enantiomeric, chiral, isomeric, ???
        self.lag = None
        self.lag_id = None # need to specify who to lag
        self.kwargs = self.__dict__

    def set_pose(self, pose):
        self.pose = pose

    def set_pars(self, pardict):
        """
        Manually update all the parameters of the Synchrony class with a configuration dictionary.
        @param pardict: Dictionary of synchrony parameters as keys and values as values.
        @type pardict: dict
        """

        self.__dict__.update(pardict)
        self.kwargs.update(pardict)

    def set_roi(self, roi):
        """
        Set the region of interest (face or body).
        @param roi: String representing face or body ('face' or 'body').
        @type roi: str
        """
        self.roi = roi

    def set_joints(self, joint_key):
        """
        Set the joints to consider for synchrony measures.
        @param joint_key: List of joint IDs from SMPL key.
        @type joint_key: list
        """
        self.joints = joint_key

    def set_tracks(self, people_of_interest):
        """
        Select which tracks to consider in synchrony calculation.
        @param people_of_interest: List of track IDs or person labels from clustering.
        @type people_of_interest: list
        """

        self.track_interest = people_of_interest

    def static(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.kwargs.update(kwargs)

        if len(self.track_interest) == 2:
            specificity = 'dyad'

        elif len(self.track_interest) < 2:
            raise ValueError('All synchrony calculations must include > 2 people.')

        if specificity=='dyad':
            trackA, trackB = self.pose.pose_data[self.track_interest[0]], self.pose.pose_data[self.track_interest[1]]
            synchrony_out = calculate_static_synchrony(trackA, trackB, frame_range=self.frame_range)
            return synchrony_out

        #elif specificity=='all':



























