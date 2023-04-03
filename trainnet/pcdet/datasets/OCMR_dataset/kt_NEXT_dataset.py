import copy
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import os
import random
import cv2
import h5py
from pcdet.utils.utils import *
import pcdet.utils.read_ocmr
import pcdet.utils.compressed_sensing as cs
from pcdet.utils.dnn_io import to_tensor_format

from pcdet.datasets.dataset import DatasetTemplate
from tools.visual_utils.visualizer import *


class kt_NEXT_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, root_path=None, save_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH))
        save_path = (Path(save_path) if save_path is not None else Path(dataset_cfg.SAVE_PATH))
        super().__init__(
            dataset_cfg=dataset_cfg, training=training, root_path=root_path,
            save_path=save_path, logger=logger
        )
        self.infos = []
        self.include_data_info(self.mode) #fill in self.info

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        data_path = self.root_path / self.infos[index]
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

################################################################
############get x_gnd,to build the undersample kspace###########
#################################################################

        nt = data_dict['kt'].shape[0]
        data_dict['coil_sens'] = np.tile(data_dict['coil_sens'][np.newaxis], (nt, 1, 1, 1))  # [nt, coil, nx, ny]

        data_dict = self.prepare_data(data_dict=data_dict)  #scale, crop

        kt = data_dict['kt']   # [nt, coil, nx, ny]
        xt = data_dict['xt']   # [nt, coil, nx, ny]
        coil_sens = data_dict['coil_sens']

        # preprocess x_gnd
        x_gnd = np.sum(coil_sens.conjugate() * xt, axis=1) / (np.linalg.norm(coil_sens, axis=1) + 1e-8)  #[nt,nx,ny]
        xf_gnd = fftshift(fft(ifftshift(x_gnd, axes=-0), norm='ortho'), axes=0)

################################################################
#########################build the undersample kspace###########
#################################################################
        ACC = self.dataset_cfg.ACC
        mask = cs.shear_grid_mask(x_gnd.shape, ACC, sample_low_freq=True, sample_n=4)

        x_und, k_und = cs.undersample(x_gnd, mask, centred=False, norm='ortho')  # [nt,nx.ny]

        data_dict = {"x_und": x_und,
                     "k_und": k_und,
                     "mask": mask,
                     "x_gnd": x_gnd,
                     "xf_gnd": xf_gnd,
                     'kt_name': self.infos[index]
                     }

        for key, value in data_dict.items():
            if key in ['kt_name']:
                continue
            mask = False
            if key in ['mask']:
                mask = True
            value = value[None]
            data_dict[key] = to_tensor_format(value, mask=mask).squeeze(0) # [2,nx,ny,nt]

        return data_dict

    def include_data_info(self, mode):
        self.logger.info('Loading dataset')
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.save_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                self.infos = infos
        self.logger.info('Total samples for dataset: %d' % (len(self.infos)))