import os
import argparse
import math
import numpy as np
import torch
import yaml
import logging
from collections import namedtuple

code_dir = os.path.abspath(__file__).replace('config.py','')
project_dir = os.path.abspath(__file__).replace('/src/lib/config.py','')
root_dir = project_dir.replace(project_dir.split('/')[-1],'')#os.path.abspath(__file__).replace('/CenterMesh/src/config.py','')
model_dir = os.path.join(project_dir,'models')
trained_model_dir = os.path.join(project_dir,'trained_models')

psypose_cfg_path = os.path.join(project_dir, '/src/configs/ROMP_config.pkl')

if torch.cuda.is_available():
    gpu_val = 0
else:
    print('No GPU detected by PyTorch. Pose estimation will be performed using CPU.')
    gpu_val = -1

ROMP_pars = {'tab': 'hrnet_cm64_single_image_test',
               'configs_yml': os.path.join(project_dir, 'src/configs/single_image.yml'),
               #'demo_image_folder': '/path/to/image_folder',
               'local_rank': 0,
               'model_version': 1,
               'multi_person': True,
               'collision_aware_centermap': False,
               'collision_factor': 0.2,
               'kp3d_format': 'smpl24',
               'eval': False,
               'max_person': 64,
               'input_size': 512,
               'Rot_type': '6D',
               'rot_dim': 6,
               'centermap_conf_thresh': 0.25,
               'centermap_size': 64,
               'deconv_num': 0,
               'model_precision': 'fp32',
               'backbone': 'hrnet',
               'gmodel_path': os.path.join(trained_model_dir, 'ROMP_hrnet32.pkl'),
               'print_freq': 50,
               'fine_tune': True,
               'gpu': '0',
               'batch_size': 64,
               'val_batch_size': 1,
               'nw': 4,
               'calc_PVE_error': False,
               'dataset_rootdir': os.path.join(root_dir,'dataset/'),
               'high_resolution': True,
               'save_best_folder': os.path.join(root_dir,'checkpoints/'),
               'log_path': os.path.join(root_dir,'log/'),
               'total_param_count': 85,
               'smpl_mean_param_path': os.path.join(model_dir,'satistic_data','neutral_smpl_mean_params.h5'),
               'smpl_model': os.path.join(model_dir,'statistic_data','neutral_smpl_with_cocoplus_reg.txt'),
               'smplx_model': True,
               'cam_dim': 3,
               'beta_dim': 10,
               'smpl_joint_num': 22,
               'smpl_model_path': os.path.join(model_dir),
               'smpl_uvmap': os.path.join(model_dir, 'smpl', 'uv_table.npy'),
               'smpl_female_texture': os.path.join(model_dir, 'smpl', 'SMPL_sampleTex_f.jpg'),
               'smpl_male_texture': os.path.join(model_dir, 'smpl', 'SMPL_sampleTex_m.jpg'),
               'smpl_J_reg_h37m_path': os.path.join(model_dir, 'smpl', 'J_regressor_h36m.npy'),
               'smpl_J_reg_extra_path': os.path.join(model_dir, 'smpl', 'J_regressor_extra.npy'),
               'kernel_sizes': [5],
               'GPUS': gpu_val,
               'use_coordmaps': True,
               'webcam': False,
               'video_or_frame': False,
               'save_visualization_on_img': False,
               'output_dir': '../demo/images_results/',
               'save_mesh': False,
               'save_centermap': False, # What is the centermap?
               'save_dict_results': False,
               'multiprocess': False}

class load_args:
    def __init__(self, **entries):
        self.__dict__.update(entries)

args = load_args(**ROMP_pars)

