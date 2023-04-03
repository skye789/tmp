from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data
from pcdet.datasets.processor.data_processor import DataProcessor

class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, training=True, root_path=None, save_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        if self.dataset_cfg is None:
            return

        self.training = training
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.save_path = save_path if save_path is not None else Path(self.dataset_cfg.SAVE_PATH)

        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.data_processor = DataProcessor(self.dataset_cfg.DATA_PROCESSOR, training=training)


    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict):

        # if self.training:
        #     if self.use_data_augmentor:
        #         data_dict = self.data_augmentor.forward(
        #             data_dict={
        #                 **data_dict,
        #                 'gt_boxes_mask': gt_boxes_mask
        #             }
        #         )

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        return data_dict