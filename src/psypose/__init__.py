#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:34:17 2021

@author: Landry Bulls
"""

__author__ = """Landry Bulls """
__email__ = 'Landry.S.Bulls@Dartmouth.edu'
__all__ = ['augment', 'data', 'display', 'extract', 'face_identification', 'features', 'pose_estimation', 'run_pare', 'tracking', 'utils']

from psypose.utils import check_data_files, check_pare_install

# On import, check for PARE installation. If not installed, offer installation. 
check_data_files()
check_pare_install()


