#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

from setuptools import setup, find_packages


#requirements_path = Path(__file__).resolve().parent.joinpath('requirements.txt')
#requirements = requirements_path.read_text().splitlines()

setup(name='psypose',
      version='0.0.1',
      description='Tools for processing pose information in psychological research.',
      url='http://github.com/scraplab/psypose',
      author='SCRAP Lab',
      author_email='Landry.S.Bulls@dartmouth.edu',
      license='MIT',
      packages=['psypose'],
      install_requires=['py-feat',
             'face-recognition',
             'nibabel',
             'networkx',
             'scikit-learn',
             'pyyaml',
             'future',
             'tqdm',
             'scipy',
             'chumpy',
             'numba',
             'yacs',
             'joblib',
             'trimesh',
             'pyrender',
             'pytube',
             'filterpy',
             'h5py',
             'scikit-image',
             'opencv-python>=3.4.11.43',
             'torch',
             'torchvision',
             'tensorboard==2.0',
             'git+https://github.com/ZhengyiLuo/smplx.git',
             'git+https://github.com/mkocabas/yolov3-pytorch.git',
             'git+https://github.com/pytorch/vision.git',
             'git+https://github.com/mkocabas/multi-person-tracker.git'],
      zip_safe=False)