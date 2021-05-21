#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

meva_cfg_files = ['psypose/MEVA/meva/cfg/train_meva_1.yml',
                  'psypose/MEVA/meva/cfg/train_meva_2.yml',
                  'psypose/MEVA/meva/cfg/train_meva.yml',
                  'psypose/MEVA/meva/cfg/train_vae.yml',
                  'psypose/MEVA/meva/cfg/vae_rec_1.yml',
                  'psypose/MEVA/meva/cfg/vae_rec_2.yml']

setup(name='psypose',
      version='0.0.1',
      description='Tools for processing pose information in psychological research.',
      url='http://github.com/scraplab/psypose',
      author='SCRAP Lab',
      author_email='Landry.S.Bulls@dartmouth.edu',
      license='MIT',
      packages=find_packages(),
      package_data=meva_cfg_files,
      install_requires=requirements,
      zip_safe=False)
