"""
Tools for formatting the ROMP pose estimation outputs.
"""

from psypose import utils
import numpy as np

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

def gather_tracks(input_data):
    trackifier = Trackifier(input_data)
    trackifier.run_sequencers()
    output_data = trackifier.tracks
    return output_data




