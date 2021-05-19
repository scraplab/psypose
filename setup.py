#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import glob
meva_files = []
dirs = glob.glob('MEVA/*')

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='psypose',
      version='0.0.1',
      description='Tools for processing pose information in psychological research.',
      url='http://github.com/scraplab/psypose',
      author='SCRAP Lab',
      author_email='Landry.S.Bulls@dartmouth.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=requirements,
      zip_safe=False)