#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:30:19 2021

@author: f004swn
"""

import numpy as np


def encode_faces(pose_object, overwrite=False, encoder='default', use_TR=False, every=None, out=None):
    if every==None:
        # encode every frame if sampling parameter not defined
        every=1
        
    def get_data(trans, parameter):
        enc = trans[parameter]
        arr = np.zeros((enc.shape[0], len(enc[0])))
        for frame in range(arr.shape[0]):
            arr[frame,:]=enc[frame]
        return arr
    print('\nParsing video data...')
    vid = VideoStim(self.vid_path)
    vid_filt = FrameSamplingFilter(every=every).transform(vid)
    frame_conv = VideoFrameCollectionIterator()
    vid_frames = frame_conv.transform(vid_filt)
    
    if torch.cuda.is_available():
        ext_loc = FaceRecognitionFaceLocationsExtractor(model='cnn')
    else:
        ext_loc = FaceRecognitionFaceLocationsExtractor()
    #for this, change to model='cnn' for GPU use on discovery

    print("\nExtracting face locations...")
    # This is using pliers, which uses dlib for extraction face locations (SOTA)
    face_locs_data = ext_loc.transform(vid_frames)
    #remove frames without faces
    face_locs_data = [obj for obj in face_locs_data if len(obj.raw) != 0]
    # convert to dataframe and remove frames with no faces

    frame_ids = [obj.stim.frame_num for obj in face_locs_data]
    locations = [obj.raw for obj in face_locs_data if len(obj.raw) != 0]
    
    # expanding multi-face frames
    frame_ids_expanded = []
    bboxes_expanded = []
    face_imgs = []
    for f, frame in enumerate(frame_ids):
        face_bboxes = locations[f]
        frame_img = utils.frame2array(frame, self.video_cv2)
        for i in face_bboxes:
            frame_ids_expanded.append(frame)
            bboxes_expanded.append(i)
            face_imgs.append(utils.crop_image(frame_img, i))
            
    bboxes = np.array(bboxes_expanded)

    print("\nExtracting face encodings...")
    
    if encoder=='default':
        encoding_length = 128
        encode = utils.default_encoding
    elif encoder=='facenet':
        encoding_length = 128
        encode = facenet_keras.encode
    elif encoder=='deepface':
        encoding_length = 2622
        encode = deepface.encode
    
    out_array = np.empty((len(frame_ids_expanded), 5+encoding_length))
    encodings = []
    for image in tqdm(face_imgs):
        encodings.append(encode(image))
    out_array[:,0], out_array[:,1:5], out_array[:, 5:] = frame_ids_expanded, bboxes, np.array(encodings)
    
    face_df = pd.DataFrame(columns=['frame_ids', 'bboxes', 'encodings'])
    face_df['frame_ids'] = frame_ids_expanded
    face_df['bboxes'] = bboxes_expanded
    face_df['encodings'] = encodings
    
    self.face_data = face_df
    
    if out != None:
        print('Saving results...')
        np.savetxt(out, out_array, delimiter=',')
    self.faces_encoded = True
