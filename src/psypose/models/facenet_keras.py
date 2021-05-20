#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 17:40:20 2021

@author: f004swn
"""

from keras.models import load_model
import numpy as np
from PIL import Image

# How to set this path so that it imports automatically? Will this work 
# if the package is pip installed?
import os
print(os.getcwd())

model_path = __file__[:-16]+'model_weights/facenet_keras.h5'
model = load_model(model_path, compile=False)
#facenet expects that the pixel values be standardized 

def encode(face_array):
    # this takes the RGB form image array that is output my the PLIERS loop
    face = Image.fromarray(face_array)
    face = face.resize((160,160))
    face = np.asarray(face)
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    samples = np.expand_dims(face, axis=0)
    yhat = model.predict(samples)
    encoding = yhat[0]
    return encoding

