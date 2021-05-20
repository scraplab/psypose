#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:34:17 2021

@author: Landry Bulls
"""

__author__ = """Landry Bulls """
__email__ = 'Landry.S.Bulls@Dartmouth.edu'
__all__ = ['extract', 'data', 'utils', 'display', 'regressors', 'MEVA']

import gdown
import os
import zipfile36 as zipfile
import shutil

return_dir = os.getcwd()
print(__file__)
os.chdir('models/model_weights')
# face encoding models:
print('Downloading facenet weights...')
gdown.download("https://drive.google.com/uc?id=1eyE-IIHpkswHhYnPXX3HByrZrSiXk00g")
print('Downloading deepface weights...')
gdown.download("https://drive.google.com/uc?id=1AkYZmHJ_LsyQYsML6k72A662-AdKwxsv")

os.chdir('../../MEVA')
os.mkdir('data')
os.chdir('data')
# MEVA files (zipped)
print('Downloading MEVA models...')
gdown.download("https://drive.google.com/uc?export=download&id=1l5pUrV5ReapGd9uaBXrGsJ9eMQcOEqmD")
with zipfile.ZipFile('meva_data.zip', 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())
os.remove('meva_data.zip')
os.chdir('..')

os.rename("data/meva_data/model_1000.p", "results/meva/vae_rec_2/models/model_1000.p")
shutil.move("data/meva_data/model_1000.p", "results/meva/vae_rec_2/models/model_1000.p")
os.replace("data/meva_data/model_1000.p", "results/meva/vae_rec_2/models/model_1000.p")


os.rename("data/meva_data/spin_model_checkpoint.pth.tar", "results/meva/train_meva_2/spin_model_checkpoint.pth.tar")
shutil.move("data/meva_data/spin_model_checkpoint.pth.tar", "results/meva/train_meva_2/spin_model_checkpoint.pth.tar")
os.replace("data/meva_data/spin_model_checkpoint.pth.tar", "results/meva/train_meva_2/spin_model_checkpoint.pth.tar")

os.chdir(return_dir)

from psypose.data import pose
from psypose.extract import annotate

