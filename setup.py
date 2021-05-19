#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

from setuptools import setup, find_packages


requirements_path = Path(__file__).resolve().dirname().joinpath('requirements.txt')
requirements = requirements_path.read_text().splitlines()

setup(name='psypose',
      version='0.0.1',
      description='Tools for processing pose information in psychological research.',
      url='http://github.com/scraplab/psypose',
      author='SCRAP Lab',
      author_email='Landry.S.Bulls@dartmouth.edu',
      license='MIT',
      packages=['psypose'],
      install_requires=requirements,
      zip_safe=False)