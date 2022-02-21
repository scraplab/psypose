#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 20:51:06 2021

@author: Landry Bulls

"""
import numpy as np
from keras.layers import Dropout, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Activation
from keras import Sequential
from keras.applications.resnet import preprocess_input
import cv2
from psypose.utils import PSYPOSE_DATA_DIR

# this model has been adopted from: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

modelfile = PSYPOSE_DATA_DIR.joinpath('vgg_face_weights.h5')

def face_model(mod_weights):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
     
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    model.load_weights(mod_weights)
    return model

vgg_model = face_model(modelfile)


def encode(array):
    # this will convert cv2 arrays into a format readable to the VGG NN
    array = cv2.resize(array, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
    array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    if array.dtype == 'uint8':
        array = array.astype(dtype='float32')
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return vgg_model.predict(array)[0,:]



    

