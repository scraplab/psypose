#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

dep_links = ['https://github.com/LandryBulls/py-feat.git',
             'https://github.com/ZhengyiLuo/smplx.git',
             'https://github.com/mkocabas/yolov3-pytorch.git',
             'https://github.com/pytorch/vision.git',
             'https://github.com/mkocabas/multi-person-tracker.git']

setup(name='psypose',
      version='0.0.1',
      description='Tools for processing pose information in psychological research.',
      url='http://github.com/scraplab/psypose',
      author='SCRAP Lab',
      author_email='Landry.S.Bulls@dartmouth.edu',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False)
