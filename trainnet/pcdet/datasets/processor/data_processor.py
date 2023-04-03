from functools import partial

import numpy as np
from skimage import transform
from pcdet.utils.utils import *

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class DataProcessor(object):
    def __init__(self, processor_configs, training):
        self.training = training
        self.mode = 'train' if training else 'test'
        self.data_processor_queue = []

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def demo_process(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.demo_process, config=config)
        return data_dict


    def scale_kt(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.scale_kt, config=config)
        scale = config.SCALE
        data_dict['kt'] *= scale
        return data_dict

    def crop_kt_coil_sens(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.crop_kt_coil_sens, config=config)
        nt, coil, nx, ny = data_dict['kt'].shape
        xt = kt2xt(data_dict['kt'])
        TC = config.CROP_TIME
        SC = config.CROP_ROW
        if TC <= 0:
            TC = nt
        if SC <= 0:
            SC = 0
        xt = xt[:TC, :,  int(nx * SC): nx-int(nx * SC), :]  # [nt, coil, nx, ny]
        data_dict['coil_sens'] = data_dict['coil_sens'][:TC, :, int(nx * SC): nx-int(nx * SC), :]
        data_dict['kt'] = img2kspace(xt)
        data_dict['xt'] = xt
        return data_dict


    def forward(self, data_dict):
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
