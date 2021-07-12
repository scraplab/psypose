import logging
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from psypose.ROMP.src.core import _init_paths
import config
import constants
from config import psypose_cfg_path
from models import build_model
import joblib
from utils import load_model, get_remove_keys, reorganize_items
from dataset.mixed_dataset import SingleDataset
from visualization.visualization import Visualizer

ROMP_config_dict = joblib.load(psypose_cfg_path)

if ROMP_config_dict['model_precision']=='fp16':
    from torch.cuda.amp import autocast, GradScaler

class Base(object):
    def __init__(self):
        self.project_dir = config.project_dir
        hparams_dict = self.load_config_dict(ROMP_config_dict)
        self._init_params()

    def _build_model_(self):
        logging.info('start building model.')
        model = build_model()
        if '-1' not in self.gpu:
            model = model.cuda()
        model = load_model(self.gmodel_path, model, \
            prefix = 'module.', drop_prefix='')
        self.model = nn.DataParallel(model)
        logging.info('finished build model.')

    def _init_params(self):
        self.global_count = 0
        self.demo_cfg = {'mode':'val', 'calc_loss': False}
        self.eval_cfg = {'mode':'train', 'calc_loss': False}
        self.gpus = [int(i) for i in self.gpu.split(',')]

    def _create_data_loader(self,train_flag=True):
        logging.info('gathering datasets')
        datasets = MixedDataset(train_flag=train_flag)
        return DataLoader(dataset = datasets,\
                batch_size = self.batch_size if train_flag else self.val_batch_size, shuffle = True, \
                drop_last = True if train_flag else False, pin_memory = True,num_workers = self.nw)

    def _create_single_data_loader(self, **kwargs):
        logging.info('gathering datasets')
        datasets = SingleDataset(**kwargs)
        return DataLoader(dataset = datasets, shuffle = False,batch_size = self.val_batch_size,\
                drop_last = False if self.eval else True, pin_memory = True, num_workers = self.nw)

    def load_config_dict(self, config_dict):
        hparams_dict = {}
        for i, j in config_dict.items():
            setattr(self,i,j)
            hparams_dict[i] = j

        logging.basicConfig(level=logging.INFO if self.local_rank in [-1, 0] else logging.WARN)
        logging.info(config_dict)
        logging.info('-'*66)
        return hparams_dict

    def net_forward(self, meta_data, cfg=None):
        ds_org, imgpath_org = get_remove_keys(meta_data,keys=['data_set','imgpath'])
        meta_data['batch_ids'] = torch.arange(len(meta_data['image']))
        if self.model_precision=='fp16':
            with autocast():
                outputs = self.model(meta_data, **cfg)
        else:
            outputs = self.model(meta_data, **cfg)

        outputs['meta_data']['data_set'], outputs['meta_data']['imgpath'] = reorganize_items([ds_org, imgpath_org], outputs['reorganize_idx'].cpu().numpy())
        return outputs