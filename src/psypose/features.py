#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Functions for the construction of second-order pose data and regressors

"""
import numpy as np
import pandas as pd
import scipy.stats
from pyquaternion import Quaternion
from psypose import utils
import os

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


static_max = np.sqrt(2)*23

def get_pose_distance(p1, p2):
    # p1 and p2 should just be the vectors
    # may be the normal static vectors or those derived with numpy.gradient()
    p1, p2 = [Quaternion(i) for i in p1[1:]], [Quaternion(i) for i in p2[1:]]
    distance = np.sum([(p1[k]-p1[k]).norm for k in range(23)])
    return distance

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

def calculate_static_synchrony(A, B, frame_range='all'):
    trackA = dict(A)
    trackB = dict(B)
    framesA, framesB = trackA['frame_ids'], trackB['frame_ids']
    if frame_range != 'all':
        assert isinstance(frame_range, tuple), 'If not "all", please input frame_range as tuple. Eg: (in_frame, out_frame).'
        framelist = np.arange(frame_range[0], frame_range[1])
        frameLocsA, frameLocsB = np.where(np.isin(framesA, framelist))[0], np.where(np.isin(framesB, framelist))[0]
        #
        # for key, value in trackA.items():
        #     trackA[key] = value[frameLocsA]
        # for key, value in trackB.items():
        #     trackB[key] = value[frameLocsB]
        framecount = len(framelist)
    else:
        framecount = len(framesA)
        assert len(trackA['frame_ids']) == len(
            trackB['frame_ids']), "Tracks are of different framecounts. Please set frame range as a tuple."
    trackA, trackB = trackA['quaternion'], trackB['quaternion']
    static_sync_timeseries = []
    for i in range(framecount):
        #pose1 = [Quaternion(j) for j in trackA[i]]
        #pose2 = [Quaternion(j) for j in trackB[i]]
        pose1, pose2 = trackA[i], trackB[i]
        distance = np.sum([Quaternion.absolute_distance(pose1[k], pose2[k]) for k in range(1,20)])
        static_sync = -(2 * (distance / max_distance_static)) + 1
        static_sync_timeseries.append(static_sync)
    return np.array(static_sync_timeseries)


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



























