import os
import argparse
import math
import numpy as np
import torch
import yaml
import logging
 
code_dir = os.path.abspath(__file__).replace('config.py','')
project_dir = os.path.abspath(__file__).replace('/src/lib/config.py','')
root_dir = project_dir.replace(project_dir.split('/')[-1],'')#os.path.abspath(__file__).replace('/CenterMesh/src/config.py','')
model_dir = os.path.join(project_dir,'models')
trained_model_dir = os.path.join(project_dir,'trained_models')

psypose_cfg_path = os.path.join(project_dir, '/src/configs/ROMP_config.pkl')
