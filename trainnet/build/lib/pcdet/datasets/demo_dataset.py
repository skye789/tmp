import copy
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import os
import random

from pcdet.datasets.dataset import DatasetTemplate

class Demo_Dataset(DatasetTemplate):
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
        data_dict = {}
        img_path = str(self.root_path / self.infos[index])
        # img = cv2.imread(img_path)
        # data_dict['images'] = img
        return data_dict

    def include_data_info(self, mode):
        self.logger.info('Loading dataset')
        infos_data = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.save_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                infos_data.extend(infos)

        self.infos.extend(infos_data)
        self.logger.info('Total samples for dataset: %d' % (len(infos_data)))


def create_dataset_info(data_path, save_path, split_percent=0.7):
    file_names = os.listdir(data_path)
    random.shuffle(file_names)
    split_index = int(split_percent*len(file_names))
    train_file_name = file_names[:split_index]
    test_file_name = file_names[split_index:]

    train_infos_pkl = 'demo_dataset_train_infos.pkl'
    test_infos_pkl = 'demo_dataset_test_infos.pkl'

    with open(save_path / train_infos_pkl, 'wb') as f:
        pickle.dump(train_file_name, f)
    with open(save_path / test_infos_pkl, 'wb') as f:
        pickle.dump(test_file_name, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--data_path', type=str, help='')
    parser.add_argument('--save_path', type=str, help='')
    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    dataset_cfg = dataset_cfg.DATA_CONFIG
    if args.data_path is not None:
        ROOT_DIR = Path(args.data_path)
    else:
        ROOT_DIR = Path(dataset_cfg.DATA_PATH)

    if args.save_path is not None:
        SAVE_DIR = Path(args.save_path)
    else:
        if dataset_cfg.get('SAVE_PATH', None) is not None:
            SAVE_DIR = Path(dataset_cfg.SAVE_PATH)
        else:
            SAVE_DIR = ROOT_DIR
    create_dataset_info(
        data_path=ROOT_DIR,
        save_path=SAVE_DIR,
    )
