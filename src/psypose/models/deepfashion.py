#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 12:20:39 2021

@author: f004swn
"""
import os
import numpy as np
from keras.layers import Dropout, ZeroPadding2D, Convolution2D, MaxPooling2D, Flatten, Activation
from keras import Sequential
from keras.applications.resnet import preprocess_input
import cv2
from tensorflow import keras
from mrcnn.model import MaskRCNN
from mrcnn.config import Config


os.chdir('/Users/f004swn/Documents/Code/packages')


modelfile = 'psypose/models/model_weights/deepfashion2.h5'


class_names = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling', 
               'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress',
               'vest_dress', 'sling_dress']

class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 13

rcnn = MaskRCNN(mode='inference', config=TestConfig())
rcnn.load_weights(modelfile, by_name=True)