import pandas as pd
import os 
from pathlib import Path
from tqdm import tqdm
import joblib
import numpy as np
from librosa.feature import rms
from scipy.signal import resample
import cv2
from scipy.io import wavfile
import librosa
import time
import argparse


# set up input args
#parser = argparse.ArgumentParser(description='Generate a people dict object for a session.

mydir = Path('/safestore/users/landry/SCRAP/data/andromeda_storage/conversations_unconstrained')
dirs = [f for f in mydir.iterdir() if f.is_dir()]
dirs = [i for i in dirs if '_' in i.name]
dirs = [i for i in dirs if len(i.name.split('_')[1]) == 3]
dirs = [i for i in dirs if (i / 'derivatives' / 'cam1_concatenated_trimmed.mp4').exists()]
dirs = [i for i in dirs if (i / 'derivatives' / 'cam2_concatenated_trimmed.mp4').exists()]

def check_pose_and_face(dir):
    p1, p2 = dir / 'processed' / 'cam1_pose.pkl', dir / 'processed' / 'cam2_pose.pkl'
    f1, f2 = dir / 'processed' / 'cam1_faces.txt', dir / 'processed' / 'cam2_faces.txt'
    if p1.exists() and p2.exists() and f1.exists() and f2.exists():
        return True
    else:
        return False

def frame2array(frame_no, vidpath):
    cap = cv2.VideoCapture(str(vidpath))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_frameid(x):
    filename = os.path.basename(x)
    frame_id = int(filename.split('_')[-1].split('.')[0].lstrip('0'))-1
    return frame_id

def merge_data(batch_folder):
    print(f'Merging face data from {batch_folder}...')
    csvs = [x for x in batch_folder.iterdir() if x.suffix == '.csv']
    csvs = sorted(csvs, key=lambda x: int(x.stem.split('_')[-1]))
    df = pd.DataFrame(columns = pd.read_csv(csvs[0]).columns)
    for csv in tqdm(csvs):
        df = pd.concat([df, pd.read_csv(csv)], ignore_index=True)
    df['frame'] = df['input'].apply(get_frameid)
    df.sort_values(by='frame', inplace=True)
    return df

def body2tblr(bbox):
    x, y, w, h = bbox
    top, bottom, left, right = [int(round(i)) for i in [y-h/2, y+h/2, x-w/2, x+w/2]]
    return top, left, bottom, right

def face2tblr(bbox):
    cx, cy, w, h = bbox
    top, bottom, left, right = [int(round(i)) for i in [cy, cy+h, cx, cx+w]]
    return top, left, bottom, right

def assign_track(FaceRectX, bboxes_dict):
    match = False
    for track, bbox in bboxes_dict.items():
        top, left, bottom, right = body2tblr(bbox)
        if left < FaceRectX < right:
            match = True
            return track
    if not match:
        return None

def match_tracks(face_data, body_data):
    face_data = face_data.copy()
    tracks = list(body_data.keys())
    bboxes = [body_data[track]['bboxes'][0] for track in tracks]
    bboxes = dict(zip(tracks, bboxes))
    face_data['track'] = face_data['FaceRectX'].apply(lambda x: assign_track(x, bboxes))
    return face_data

def clean_duplicate_frames(df):
    frames = df['frame'].unique()
    occ = {frame: len(df[df['frame'] == frame]) for frame in frames}
    duplicates = {k: v for k, v in occ.items() if v > 1}
    for frame in duplicates.keys():
        dup = df[df['frame'] == frame]
        maxscore = dup['FaceScore'].max()
        df = df.drop(dup[dup['FaceScore'] < maxscore].index, inplace=False)
    return df

def interpolate_face_data(face_df, framecount, limit=60):
    face_df = face_df.copy()
    fidx = pd.Index(range(framecount))
    face_df = face_df.set_index('frame').reindex(fidx).reset_index()
    face_df = face_df.interpolate(method='linear', limit=limit)
    face_df.fillna(method='ffill', inplace=True)
    return face_df

def body2tblr(bbox):
    x, y, w, h = bbox
    top, bottom, left, right = [int(round(i)) for i in [y-h/2, y+h/2, x-w/2, x+w/2]]
    return top, left, bottom, right

def face2tblr(bbox):
    cx, cy, w, h = bbox
    top, bottom, left, right = [int(round(i)) for i in [cy, cy+h, cx, cx+w]]
    return top, left, bottom, right

def avg_id(df, threshold=0.001):
    idcols = [i for i in df.columns if 'Identity_' in i]
    df = df.copy()
    df = df[df['FaceScore'] > threshold]
    return df[idcols].mean(axis=0).to_numpy()

def generate_id_matrix(people):
    # assumes that the people dict has been generated but not matched to audio
    people = people.copy()
    cam1_tracks, cam2_tracks = {k: v for k, v in people.items() if v['cam'] == 1}, {k: v for k, v in people.items() if v['cam'] == 2}
    cam1_id_vecs, cam2_id_vecs = [], []
    for track in cam1_tracks.keys():
        cam1_id_vecs.append(avg_id(cam1_tracks[track]['face']))
    for track in cam2_tracks.keys():
        cam2_id_vecs.append(avg_id(cam2_tracks[track]['face']))
    cam1_id_vecs, cam2_id_vecs = np.array(cam1_id_vecs), np.array(cam2_id_vecs)
    corrmat = np.corrcoef(cam1_id_vecs, cam2_id_vecs)[len(cam1_id_vecs):, :len(cam1_id_vecs)]
    # rows are cam1, columns are cam2
    return corrmat

def resolve_duplicates(ppl):
    # operates on an umatched people dict
    ##### for now, this only consolidates two tracks (i.e., assumes only one person is redundant across the two cameras)
    people = ppl.copy()
    cam1_tracks, cam2_tracks = {k: v for k, v in people.items() if v['cam'] == 1}, {k: v for k, v in people.items() if v['cam'] == 2}
    cam1_id_vecs, cam2_id_vecs = [], []
    for track in cam1_tracks.keys():
        cam1_id_vecs.append(avg_id(cam1_tracks[track]['face']))
    for track in cam2_tracks.keys():
        cam2_id_vecs.append(avg_id(cam2_tracks[track]['face']))
    cam1_id_vecs, cam2_id_vecs = np.array(cam1_id_vecs), np.array(cam2_id_vecs)
    corrmat = np.corrcoef(cam1_id_vecs, cam2_id_vecs)[len(cam1_id_vecs):, :len(cam1_id_vecs)]

    idx = np.unravel_index(np.argmax(corrmat, axis=None), corrmat.shape)
    # the two tracks to consolidate
    track1, track2 = list(cam1_tracks.keys())[idx[1]], list(cam2_tracks.keys())[idx[0]]

    # consolidate the two tracks
    people[track1]['face2'] = people[track2]['face']
    people[track1]['body2'] = people[track2]['body']
    people[track1]['vpath2'] = people[track2]['vpath']
    people[track1]['cam2'] = people[track2]['cam']
    people[track1]['track2'] = people[track2]['track']

    # remove the second track
    people.pop(track2)
    return people

# function for returning a dict of people/tracks that include face and body as values, each of which are themselves dicts
def get_people_dict(session_path):
    session_path = Path(session_path)
    naud = len(os.listdir(session_path / 'audio'))
    
    print('Loading data to generate unmatched people dict...')
    cam1_faces = pd.read_csv(session_path / 'processed' / 'cam1_faces_merged.csv')
    cam1_pose = joblib.load(session_path / 'processed' / 'cam1_pose.pkl')
    cam2_faces = pd.read_csv(session_path / 'processed' / 'cam2_faces_merged.csv')
    cam2_pose = joblib.load(session_path / 'processed' / 'cam2_pose.pkl')

    cam1path = session_path / 'derivatives' / 'cam1_concatenated_trimmed_ds540.mp4'
    cam2path = session_path / 'derivatives' / 'cam2_concatenated_trimmed_ds540.mp4'

    print('Matching tracks...')
    cam1_faces = match_tracks(cam1_faces, cam1_pose)
    cam2_faces = match_tracks(cam2_faces, cam2_pose)

    # get highest framecount
    bodyframes1, bodyframes2 = max([cam1_pose[track]['frame_ids'][-1] for track in cam1_pose.keys()]), max([cam2_pose[track]['frame_ids'][-1] for track in cam2_pose.keys()])
    faceframes1, faceframes2 = max(cam1_faces['frame'].unique()), max(cam2_faces['frame'].unique())
    framecount = max(bodyframes1, bodyframes2, faceframes1, faceframes2)

    cam1_people, cam2_people = {}, {}

    print('Consolidating data...')
    for track in cam1_pose.keys():
        if track in cam1_faces['track'].values:
            face = cam1_faces[cam1_faces['track'] == track]
            face = clean_duplicate_frames(face)
            face = interpolate_face_data(face, framecount)
            body = cam1_pose[track]
            cam1_people[track] = {'face': face, 'body': body, 'cam': 1, 'vpath': cam1path}

    for track in cam2_pose.keys():
        if track in cam2_faces['track'].values:
            face = cam2_faces[cam2_faces['track'] == track]
            face = clean_duplicate_frames(face)
            face = interpolate_face_data(face, framecount)
            body = cam2_pose[track]
            cam2_people[track] = {'face': face, 'body': body, 'cam': 2, 'vpath': cam2path}

    # assign new track numbers to the people for consolidation
    people = {}

    print('Consolidating people...')
    for i, (track, data) in enumerate(cam1_people.items()):
        people[i] = data
        people[i]['track'] = i

    for i, (track, data) in enumerate(cam2_people.items()):
        people[i+len(cam1_people)] = data
        people[i+len(cam1_people)]['track'] = i+len(cam1_people)


    if naud < len(people):
        print(f'Duplicate tracks detected. Resolving duplicates for session {session_path.name}...')
        people = resolve_duplicates(people)
        
    return people

def pad_image(img, target_shape=(60,60)):
    h, w = img.shape[:2]
    target_h, target_w = target_shape
    pad_h = target_h - h
    pad_w = target_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def upscale(image, landmarks, target=(200,200)):
    orig_shape = image.shape[:2]
    image = cv2.resize(image, target)
    x, y = zip(*landmarks)
    x = np.array(x) / orig_shape[1] * target[1]
    y = np.array(y) / orig_shape[0] * target[0]
    landmarks = list(zip(x, y))
    return image, landmarks

def sort_audio(audlist):
    nums = [str(x).split('/')[-1].split('_')[0] for x in audlist]
    nums = [int(x) for x in nums]
    audlist = [x for _, x in sorted(zip(nums, audlist))]
    return audlist

def get_audio_files(session_path):
    session_path = Path(session_path)
    audio_files = [x for x in (session_path / 'processed').iterdir() if x.suffix == '.wav']
    audio_files = sort_audio(audio_files)
    return audio_files

def interp_to_vid(rms_array, framecount):
    original_indices = np.linspace(0, len(rms_array)-1, len(rms_array))
    target_indices = np.linspace(0, len(rms_array)-1, framecount)
    interp = np.interp(target_indices, original_indices, rms_array)
    return interp

def gen_rms(rms_vec, framecount):
    rms_vec = rms(y=rms_vec, frame_length=2048, hop_length=512).squeeze()
    rms_vec = interp_to_vid(rms_vec, framecount)
    return rms_vec

def get_rms_corr(rms_vec, au_vec):
    rms_vec = rms_vec[~np.isnan(au_vec)]
    au_vec = au_vec[~np.isnan(au_vec)]
    return np.corrcoef(au_vec, rms_vec)[0,1]

def assign_audio(people, audio_files):
    people = people.copy()
    audio_files = sort_audio(audio_files)
    session = people[0]['vpath'].parent.parent
    n_aud = len(audio_files)
    letters = ['A', 'B', 'C', 'D'][:n_aud]
    framecount = people[0]['face'].shape[0]
    rms_vecs = [gen_rms(wavfile.read(x)[1], framecount) for x in audio_files]
    rms_vecs = dict(zip(letters, rms_vecs))
    people_new = {}
    AU_vecs = []
    for track in people.keys():
        vec = people[track]['face']['AU26'].to_numpy()
        AU_vecs.append(vec)
    AU_vecs = np.array(AU_vecs)
    corrmat = np.zeros((len(letters), len(people.keys())))
    for i, letter in enumerate(letters):
        for j, track in enumerate(people.keys()):
            corrmat[i,j] = get_rms_corr(rms_vecs[letter], AU_vecs[j])

    for i in range(len(people.keys())):
        maxcorr = np.unravel_index(np.argmax(corrmat, axis=None), corrmat.shape)
        track_idx, letter_idx = maxcorr[1], maxcorr[0]
        track, letter = list(people.keys())[track_idx], letters[letter_idx]
        people_new[letter] = people[track]
        corrmat[letter_idx, :] = -1
        corrmat[:, track_idx] = -1

    for i, letter in enumerate(letters):
        people_new[letter]['audio_path'] = audio_files[i]
        people_new[letter]['transcript_path'] = session / 'processed' / f'{letter}_transcript.pkl'

    return people_new
            

def generate_labeled_image(people_dict_assigned, frame='auto'):
    people = people_dict_assigned.copy()
    people = {k: v for k, v in sorted(people.items(), key=lambda item: item[1]['track'])}
    framecount = people['A']['face'].shape[0]
    if frame == 'auto':
        frame = framecount // 2
    campaths = list(set([v['vpath'] for v in people.values()]))
    cam1, cam2 = [x for x in campaths if 'cam1' in str(x)][0], [x for x in campaths if 'cam2' in str(x)][0]
    cam1_people, cam2_people = {k: v for k, v in people.items() if v['cam'] == 1}, {k: v for k, v in people.items() if v['cam'] == 2}
    vpaths = [v['vpath'] for v in people.values()]
    cam1_vpath, cam2_vpath = str([v for v in vpaths if 'cam1' in str(v)][0].name), str([v for v in vpaths if 'cam2' in str(v)][0].name)

    for letter, person in people.items():
        if 'cam2' in person.keys():
            add_person = person.copy()
            add_person['face'] = add_person['face2']
            del add_person['face2']
            add_person['body'] = add_person['body2']
            del add_person['body2']
            add_person['cam'] = add_person['cam2']
            del add_person['cam2']
            add_person['vpath'] = add_person['vpath2']
            del add_person['vpath2']
            add_person['track'] = add_person['track2']
            del add_person['track2']

            if add_person['cam'] == 1:
                cam1_people[letter] = add_person
            elif add_person['cam'] == 2:
                cam2_people[letter] = add_person

    cam1_image_shape, cam2_image_shape = frame2array(0, cam1).shape, frame2array(0, cam2).shape
    assert cam1_image_shape == cam2_image_shape

    # make the cam1 image
    cam1_img = frame2array(frame, cam1)
    # add face and body bboxes, label with letter, add transcript and audio paths
    
    for letter, person in cam1_people.items():
        face = person['face'].iloc[frame]
        body = person['body']
        face_bbox = face2tblr(face[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']])
        body_bbox = body2tblr(body['bboxes'][frame])
        cam1_img = cv2.rectangle(cam1_img, (face_bbox[1], face_bbox[0]), (face_bbox[3], face_bbox[2]), (0, 255, 0), 2)
        cam1_img = cv2.rectangle(cam1_img, (body_bbox[1], body_bbox[0]), (body_bbox[3], body_bbox[2]), (0, 255, 0), 2)
        cam1_img = cv2.putText(cam1_img, letter, (face_bbox[1], face_bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        cam1_img = cv2.putText(cam1_img, Path(person['transcript_path']).name, (face_bbox[1]-50, face_bbox[2]+200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cam1_img = cv2.putText(cam1_img, Path(person['audio_path']).name, (face_bbox[1]-50, face_bbox[2]+250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # label the camera at the bottom with the video path
    cam1_img = cv2.putText(cam1_img, cam1_vpath, (10, cam1_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # make the cam2 image
    cam2_img = frame2array(frame, cam2)

    # add face and body bboxes, label with letter, add transcript and audio paths
    for letter, person in cam2_people.items():
        face = person['face'].iloc[frame]
        body = person['body']
        face_bbox = face2tblr(face[['FaceRectX', 'FaceRectY', 'FaceRectWidth', 'FaceRectHeight']])
        body_bbox = body2tblr(body['bboxes'][frame])
        cam2_img = cv2.rectangle(cam2_img, (face_bbox[1], face_bbox[0]), (face_bbox[3], face_bbox[2]), (0, 255, 0), 2)
        cam2_img = cv2.rectangle(cam2_img, (body_bbox[1], body_bbox[0]), (body_bbox[3], body_bbox[2]), (0, 255, 0), 2)
        cam2_img = cv2.putText(cam2_img, letter, (face_bbox[1], face_bbox[0]), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
        cam2_img = cv2.putText(cam2_img, Path(person['transcript_path']).name, (face_bbox[1]-50, face_bbox[2]+200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cam2_img = cv2.putText(cam2_img, Path(person['audio_path']).name, (face_bbox[1]-50, face_bbox[2]+250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # label the camera at the bottom with the video path
    cam2_img = cv2.putText(cam2_img, cam2_vpath, (10, cam2_img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)    

    # concatenate the images horizontally
    img = np.concatenate([cam1_img, cam2_img], axis=1)
    return img

def process_session(session):
    ### Merging the face batches and saving them
    session = Path(session)

    if not (session / 'processed' / 'cam1_faces_merged.csv').exists() or not (session / 'processed' / 'cam2_faces_merged.csv').exists():
        cam1_faces = merge_data(session / 'processed' / 'cam1_face_batches')
        cam2_faces = merge_data(session / 'processed' / 'cam2_face_batches')
        cam1_faces.to_csv(session / 'processed' / 'cam1_faces_merged.csv', index=False)
        cam2_faces.to_csv(session / 'processed' / 'cam2_faces_merged.csv', index=False)
    else:
        print('Face data already merged. Loading face data...')
        cam1_faces = pd.read_csv(session / 'processed' / 'cam1_faces_merged.csv')
        cam2_faces = pd.read_csv(session / 'processed' / 'cam2_faces_merged.csv')

    ### Matching poses to faces and processing face data to make it continuous (running interpolation, etc.)
    people = get_people_dict(session)

    ### Assigning audio to people
    audio_files = get_audio_files(session)
    people = assign_audio(people, audio_files)
    joblib.dump(people, str(session / 'processed' / 'people.pkl'))

    ### Generating labeled images
    img = generate_labeled_image(people)
    cv2.imwrite(str(session / 'data_assignments.jpg'), img)


# ### Running loop over all eligibile sessions
dirs = [i for i in dirs if check_pose_and_face(i)]
# ignore if people.pkl already in processed
dirs = [i for i in dirs if not (i / 'processed' / 'people.pkl').exists() and not (i / 'data_assignments.jpg').exists()]

# adding 2023-10-04_000 back in after manual fixing...
# 2023-11-03_000 is the one with inconsistent frame rates...

removes = ['2023-11-03_000', '2024-01-26_000', '2023-11-06_000']
dirs = [i for i in dirs if not any([r in i.name for r in removes])]

run = input(f'Processing {len(dirs)} sessions. Continue? (y/n): ')
if run == 'n':
    print('Exiting...')
    exit()
    
elif run == 'y':
    print(f'Processing {len(dirs)} sessions...')
    for session in tqdm(dirs):
        print(f'Processing {session.name}...')
        t0 = time.time()
        process_session(session)
        print(f'Processed {session.name} in {time.time()-t0} seconds.')






