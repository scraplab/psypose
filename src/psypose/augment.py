"""
Tools for formatting the ROMP pose estimation outputs, including track-stitching and quaternion calculation.
"""

from psypose import utils
import numpy as np
from quaternion import as_quat_array
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Rotation as R


def get_position_vec(body):
    z, x, y = body['cam'][0], body['pj2d'][0][0], body['pj2d'][0][1]
    body3d = body['j3d_smpl24']
    b3d_global = body3d + [x, y, z]
    return b3d_global.flatten()


def extract_bbox(joints2d):
    # stored as cx, cy, w, h
    cx, cy = np.min(joints2d[:, 0]), np.max(joints2d[:, 1])
    w = np.max(joints2d[:, 0]) - np.min(joints2d[:, 0])
    h = np.max(joints2d[:, 1]) - np.min(joints2d[:, 1])
    return [cx, cy, w, h]


class Trackifier(object):

    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.cur_frame = 0
        self.tracks = {}
        self.max_frame = len(reference_data) - 1
        self.threshold = 7.5
        self.track_counter = -1

    def grab_next_track_ID(self):
        self.track_counter += 1
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
        if len(self.active_sequencers) < n_candidates:
            taken_indices = [i.idx for i in self.active_sequencers]
            lonely_indices = [i for i in range(n_candidates) if i not in taken_indices]
            for idx in lonely_indices:
                self.active_sequencers.append(Sequencer(self.cur_frame, idx, self))

    def run_sequencers(self):
        len_init = len(self.reference_data[self.cur_frame])
        self.active_sequencers = [Sequencer(self.cur_frame, i, self) for i in range(len_init)]
        pbar = tqdm(total=self.max_frame)
        while self.cur_frame + 1 < self.max_frame:
            self.cur_frame += 1
            self.advance_set()
            pbar.update(1)
        for sequencer in self.active_sequencers:
            sequencer.kill()


class Sequencer(object):

    def __init__(self, init_frame, body_idx, command):
        self.is_active = True
        self.command = command
        self.cur_frame = init_frame
        self.reference_data = command.reference_data.copy()
        self.current_value = get_position_vec(self.reference_data[init_frame][body_idx])
        self.cur_body = self.reference_data[self.cur_frame][body_idx]
        self.threshold = self.command.threshold
        self.frame_ids = [init_frame]
        self.track_ID = self.command.grab_next_track_ID()
        self.track = [self.reference_data[init_frame][body_idx]]

    def reformat_track(self, in_track):
        out_track = {}
        length = len(self.frame_ids)
        out_track.update({'frame_ids': np.array(self.frame_ids)})
        out_track.update({'cam': np.array([self.track[i]['cam'] for i in range(length)])})
        out_track.update({'pose': np.array([self.track[i]['pose'] for i in range(length)])})
        out_track.update({'j3d_smpl24': np.array([self.track[i]['j3d_smpl24'] for i in range(length)])})
        out_track.update({'j3d_spin24': np.array([self.track[i]['j3d_spin24'] for i in range(length)])})
        out_track.update({'pj2d': np.array([self.track[i]['pj2d'] for i in range(length)])})
        out_track.update({'pj2d_org': np.array([self.track[i]['pj2d_org'] for i in range(length)])})
        out_track.update({'bboxes': np.array([extract_bbox(value) for value in out_track['pj2d_org']])})
        out_track = {self.track_ID: out_track}
        return out_track

    def kill(self):
        self.is_active = False
        self.command.tracks.update(self.reformat_track(self.track))

    def advance(self):
        optional_bodies = self.reference_data[self.cur_frame + 1]
        deltas = [euclidean(get_position_vec(self.cur_body), get_position_vec(ob)) for ob in optional_bodies]
        deltas_filtered = np.array([i for i in deltas if i <= self.threshold])
        candidate_indices = np.where(np.array(deltas) <= self.threshold)[0]
        if not len(deltas_filtered):
            self.kill()
        else:
            selection = np.min(deltas_filtered)
            self.idx = candidate_indices[np.where(deltas_filtered == selection)[0][0]]
            self.track.append(optional_bodies[self.idx])
            self.cur_frame += 1
            if self.cur_frame < self.command.max_frame:
                self.frame_ids.append(self.cur_frame)
            else:
                self.kill()


def gather_tracks(input_data):
    trackifier = Trackifier(input_data)
    trackifier.run_sequencers()
    output_data = trackifier.tracks
    return output_data

def add_quaternion(pose_dat):
    """
    Adds quaternion representation to pose data.
    @param pose_dat: Pose data object
    @type pose_dat: dict
    @return: pose_dat
    @rtype: dict
    """
    # PARE represents each joint as a rotation matrix (docs say otherwise but oh well)
    for track, data in pose_dat.items():
        n_frames = len(data['pose'])
        quats = np.empty((n_frames,24,4))
        for frame in range(n_frames):
            quats[frame] = np.array([R.from_matrix(data['pose'][frame][joint]).as_quat() for joint in range(24)])
        pose_dat[track].update({'quaternion':as_quat_array(quats)})
    return pose_dat

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

track_keys = ['cam', 'pose', 'j3d_smpl24', 'j3d_spin24', 'j3d_op25', 'pj2d', 'pj2d_org', 'trans']

def track_template(frame_ids):
    fc = len(frame_ids)
    track_template = {
        'frame_ids': frame_ids,
        'cam': np.empty((fc, 3)),
        'pose': np.empty((fc, 72)),
        'j3d_smpl24': np.empty((fc, 45, 3)),
        'j3d_spin24': np.empty((fc, 24, 3)),
        'j3d_op25': np.empty((fc, 25, 3)),
        'pj2d': np.empty((fc, 54, 2)),
        'pj2d_org': np.empty((fc, 54, 2)),
        'trans': np.empty((fc, 3)),
        'fill': np.array([np.nan for i in range(fc)])
    }
    for k in track_keys:
        track_template[k][:] = np.nan

    return track_template

def add_romp_frame(frame_id, body, final_dict):
    track = body['track_id']
    frameLoc = np.where(final_dict[track]['frame_ids']==frame_id)[0][0]
    final_dict[track]['fill'][frameLoc] = 1
    for k in track_keys:
        final_dict[track][k][frameLoc] = body[k]

def check_id(mpt_frame):
    return np.array([int(i[-1]) for i in mpt_frame])

def fuse_bboxes(pose):
    romp = pose.pose_data
    mpt = pose.mpt
    track_detections = []
    for i in mpt:
        for j in i:
            track_detections.append(int(j[-1]))
    all_tracks = np.unique(track_detections)
    track_frame_ids = {}
    for track in all_tracks:
        track_frame_ids.update({track:np.array([i for i in range(pose.framecount) if track in check_id(mpt[i])])})
    # all tracks is a list of the unqiue track ID's
    del track_detections
    # usable tracks are those that have a detected body in them.
    usable_tracks = []
    for f in range(pose.framecount):
        # get data from both romp and mpt for each frame
        bodies = romp[f]
        bboxes = mpt[f]
        for body in bodies:
            track_id = 'no_match'
            j2d = body['pj2d_org']
            for box in bboxes:
                bbox = box[:-1]
                if check_body_match(j2d, bbox):
                    track_id = box[-1]
                    break
            body['track_id'] = track_id
            if track_id != 'no_match':
                usable_tracks.append(track_id)
    output_dict = {}
    for track, ids in track_frame_ids.items():
        output_dict.update({track:track_template(ids)})
    for f, frame in romp.items():
        for body in frame:
            if body['track_id'] != 'no_match':
                add_romp_frame(f, body, output_dict)
    return output_dict






