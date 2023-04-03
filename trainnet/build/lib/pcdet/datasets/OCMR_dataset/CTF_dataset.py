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
# import read_ocmr

from pcdet.datasets.dataset import DatasetTemplate
from tools.visual_utils.visualizer import *


class OCMR_Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, training=True, root_path=None, save_path=None, logger=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH))
        save_path = (Path(save_path) if save_path is not None else Path(dataset_cfg.SAVE_PATH))
        super().__init__(
            dataset_cfg=dataset_cfg, training=training, root_path=root_path,
            save_path=save_path, logger=logger
        )
        self.infos = []
        self.include_data_info(self.mode)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    def __getitem__(self, index):
        data_path = self.save_path / self.infos[index]
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        nt = data_dict['kt'].shape[0]
        data_dict['coil_sens'] = np.tile(data_dict['coil_sens'][np.newaxis], (nt, 1, 1, 1))  # [nt, coil, nx, ny]

        data_dict = self.prepare_data(data_dict=data_dict)  #scale, crop

        kt = data_dict['kt']   # [nt, coil, nx, ny]
        xt = data_dict['xt']
        coil_sens = data_dict['coil_sens']  #[nt, coil, nx, ny]

        nt, ncoil, nx, ny = kt.shape
        ACC = self.dataset_cfg.ACC
        mask = np.zeros((nt, nx, ny), dtype=bool)  # nt, nx, ny
        for i in range(ACC):
            mask[i::ACC, i::ACC, :] = True  # undersample in nt and nx
        mask = mask.astype(np.int16)
        mask = np.tile(mask[:, np.newaxis], (1, ncoil, 1, 1))  # [nt, nCoil, nx, ny]

        '''undersample data, combined with coil sensitivity, sum up coil dimention, get undersampled xt'''
        k_und = fft2(xt, axes=(-2, -1), norm='ortho') * mask
        x_und = np.sum(ifft2(k_und, axes=(-2, -1), norm='ortho') * np.conj(coil_sens),
                                    axis=1) / (np.linalg.norm(coil_sens, axis=1) + 1e-8)
        # preprocess x_gnd
        x_gnd = np.sum(coil_sens.conjugate() * xt, axis=1) / (np.linalg.norm(coil_sens, axis=1) + 1e-8)

        x_und = np.array([np.real(x_und), np.imag(x_und)], dtype=np.float32)  # [2, nt, nx, ny]
        k_und = np.array([np.real(k_und), np.imag(k_und)], dtype=np.float32)  # [2, nt, nCoil, nx, ny]
        mask = np.array([mask, mask], dtype=np.float32)  # [2, nt, nCoil, nx, ny]
        x_gnd = np.array([np.real(x_gnd), np.imag(x_gnd)], dtype=np.float32)  # [2, nt, nx, ny]
        coil_sens = np.array([np.real(coil_sens), np.imag(coil_sens)], dtype=np.float32)  # [2, nt, nCoil, nx, ny]

        data_dict = {'x_und': x_und,
                     'x_gnd': x_gnd,
                     'k_und': k_und,
                     'mask': mask,
                     'coil_sens': coil_sens,
                     'kt_name': self.infos[index]}

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