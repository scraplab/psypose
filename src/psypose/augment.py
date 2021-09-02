"""
Tools for formatting the ROMP pose estimation outputs, including track-stitching and quaternion calculation.
"""

from psypose import utils
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation as R

def get_position(body):
    z, x, y = body['cam'][0], body['pj2d'][0][0], body['pj2d'][0][1]
    return (z, x, y)

def extract_bbox(joints2d):
    # stored as cx, cy, w, h
    cx, cy = np.min(joints2d[:,0]), np.max(joints2d[:,1])
    w = np.max(joints2d[:,0]) - np.min(joints2d[:,0])
    h = np.max(joints2d[:,1]) - np.min(joints2d[:,1])
    return [cx, cy, w, h]

class Trackifier(object):

    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.cur_frame = 0
        self.tracks = {}
        self.max_frame = len(reference_data)-1
        self.threshold = 0.75
        self.track_counter = -1

    def grab_next_track_ID(self):
        self.track_counter+=1
        return self.track_counter

    def set_max_frame(self, value):
        self.max_frame = value

    def set_threshold(self, value):
        self.threshold = value

    def advance_set(self):
        for sequencer in self.active_sequencers:
            sequencer.advance()
        self.active_sequencers = [i for i in self.active_sequencers if i.is_active]
        n_candidates = len(self.reference_data[self.cur_frame])
        if len(self.active_sequencers)<n_candidates:
            taken_indices = [i.idx for i in self.active_sequencers]
            lonely_indices = [i for i in range(n_candidates) if i not in taken_indices]
            for idx in lonely_indices:
                self.active_sequencers.append(Sequencer(self.cur_frame, idx, self))

    def run_sequencers(self):
        len_init = len(self.reference_data[self.cur_frame])
        self.active_sequencers = [Sequencer(self.cur_frame, i, self) for i in range(len_init)]
        while self.cur_frame+1<self.max_frame:
            self.cur_frame+=1
            self.advance_set()
        for sequencer in self.active_sequencers:
            sequencer.kill()

class Sequencer(object):

    def __init__(self, init_frame, body_idx, command):
        self.is_active = True
        self.command = command
        self.cur_frame = init_frame
        self.reference_data = command.reference_data.copy()
        self.current_value = get_position(self.reference_data[init_frame][body_idx])
        self.cur_body = self.reference_data[self.cur_frame][body_idx]
        self.threshold = self.command.threshold
        self.frame_ids = [init_frame]
        self.track_ID = self.command.grab_next_track_ID()
        self.track = [self.reference_data[init_frame][body_idx]]

    def reformat_track(self, in_track):
        out_track = {}
        length = len(self.frame_ids)
        out_track.update({'frame_ids':np.array(self.frame_ids)})
        out_track.update({'cam':np.array([self.track[i]['cam'] for i in range(length)])})
        out_track.update({'pose':np.array([self.track[i]['pose'] for i in range(length)])})
        out_track.update({'j3d_smpl24':np.array([self.track[i]['j3d_smpl24'] for i in range(length)])})
        out_track.update({'j3d_spin24':np.array([self.track[i]['j3d_spin24'] for i in range(length)])})
        out_track.update({'pj2d':np.array([self.track[i]['pj2d'] for i in range(length)])})
        out_track.update({'pj2d_org':np.array([self.track[i]['pj2d_org'] for i in range(length)])})
        out_track.update({'bboxes':np.array([extract_bbox(value) for value in out_track['pj2d_org']])})
        out_track = {self.track_ID:out_track}
        return out_track

    def kill(self):
        self.is_active=False
        self.command.tracks.update(self.reformat_track(self.track))

    def advance(self):
        optional_bodies = self.reference_data[self.cur_frame+1]
        deltas = [euclidean(get_position(self.cur_body), get_position(ob)) for ob in optional_bodies]
        deltas_filtered = np.array([i for i in deltas if i <= self.threshold])
        candidate_indices = np.where(np.array(deltas) <= self.threshold)[0]
        if not len(deltas_filtered):
            self.kill()
        else:
            selection = np.min(deltas_filtered)
            self.idx = candidate_indices[np.where(deltas_filtered==selection)[0][0]]
            self.track.append(optional_bodies[self.idx])
            self.cur_frame+=1
            if self.cur_frame < self.command.max_frame:
                self.frame_ids.append(self.cur_frame)
            else:
                self.kill()

def add_quaternion(pose_dat):
    for track, data in pose_dat.items():
        n_frames = len(data['pose'])
        quats = np.empty((n_frames,24,4))
        for i, pose_vec in enumerate(data['pose']):
            pose_vec = pose_vec.reshape((24,3))
            quats[i] = R.from_rotvec(pose_vec).as_quat()
        pose_dat[track].update({'quaternion':quats})
    return pose_dat

def gather_tracks(input_data):
    trackifier = Trackifier(input_data)
    trackifier.run_sequencers()
    output_data = trackifier.tracks
    return output_data

################## One Euro Filter ####################

class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
      s = value
    else:
      s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s

class OneEuroFilter:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    cutoff = self.mincutoff + self.beta * np.abs(edx)
    return self.x_filter.process(x, self.compute_alpha(cutoff))

#function to apply to a 1d vector
def smooth_vector(array):
    out = []
    filt = OneEuroFilter()
    for x in array:
        out.append(filt.process(x))
    return np.array(out)

# function to apply to one key (array) of pose object
def apply_one_euro(array):
    shape = array.shape
    smoothed = np.empty(shape)
    if len(shape)==2:
        for element in range(shape[1]):
            smoothed[:,element]=smooth_vector(array[:,element])
    elif len(shape)==3:
        for element in range(shape[1]):
            for vec in range(shape[2]):
                smoothed[:,element,vec] = smooth_vector(array[:,element,vec])
    return smoothed

# function to apply to pose object as a whole
def smooth_pose_data(pose_data):
    for track, data in pose_data.items():
        keys = list(data.keys())
        keys.remove('frame_ids')
        for key in keys:
            pose_data[track][key] = apply_one_euro(pose_data[track][key])
    return pose_data





