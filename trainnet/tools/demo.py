import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tools.visual_utils.viewer.viewer import Viewer

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        data_dict = {}
        frame_size = self.dataset_cfg.frame_size
        if index + frame_size >= len(self.sample_file_list):
            index = len(self.sample_file_list) - frame_size
        for i in range(frame_size):
            frame_index = index + i
            single_data_dict = self.preprocess_single_frame(frame_index)
            for key, val in single_data_dict.items():
                if isinstance(val, np.ndarray):
                    if key in data_dict:
                        data_dict[key] = np.concatenate((data_dict[key], single_data_dict[key]), axis=0)
                    else:
                        data_dict[key] = single_data_dict[key]
                else:
                    data_dict[key] = single_data_dict[key]
        data_dict['frame_size'] = frame_size
        return data_dict

    def preprocess_single_frame(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    viewer = Viewer(visual_dir='/home/liu/Endtoend/OpenPCDet/demo_res')
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            visual_list = model.forward(data_dict)

            if visual_list is not None and viewer is not None:
                for frame_idx, visual_dict in enumerate(visual_list):
                    for points, radius, color in visual_dict['points']:
                        viewer.add_points(points, radius=radius, color=color)

                    for boxes, label, scores, ids, color in visual_dict['boxes']:
                        if boxes is not None and len(boxes) > 0:
                            viewer.add_3D_boxes(boxes,
                                                color=color,
                                                line_alpha=0.5,
                                                box_info=scores,
                                                caption_size=(0.02, 0.02),
                                                ids=ids, )
                    # viewer.show_BEV(show_img=True,
                    #                 visual_info='iter_{}_batch_{}'.format(int(idx), frame_idx))
                    viewer.show_3D(sleep_time=1)
                    # viewer.clear_3D_boxes()

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes']
            # )

            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
