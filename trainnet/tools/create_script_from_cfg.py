import argparse
import datetime
import glob
import os
from pathlib import Path
import shutil

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[0:-2])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def read_file(file):
    with open(file, encoding='UTF-8') as f:
        read_all = f.read()
        f.close()
    return read_all


def rewrite_file(file, data):
    with open(file, 'w', encoding='UTF-8') as f:
        f.write(data)
        f.close()


def replace(file, old_content, new_content):
    content = read_file(file)
    content = content.replace(old_content, new_content)
    rewrite_file(file, content)


def insert(file, old_content, new_content):
    content = read_file(file)
    content = content.replace(old_content, old_content + '\n' + new_content)
    rewrite_file(file, content)


def create_dataset(cfg):
    if not cfg.get('DATA_CONFIG.DATASET'):
        return
    class_name = cfg.DATA_CONFIG.DATASET
    module_dir = Path(cfg.EXP_GROUP_PATH) / Path('pcdet/datasets')
    file_name = str.lower(class_name)
    file_path = Path(module_dir) / Path(file_name + '.py')
    demo_path = Path(module_dir) / Path('demo_dataset.py')
    init_path = Path(module_dir) / Path('__init__.py')
    if os.path.exists(file_path):
        return
    else:
        shutil.copyfile(demo_path, file_path)
        replace(file_path, 'Demo_Dataset', class_name)
        import_dataset = 'from .' + file_name + ' import ' + class_name
        import_dataset_all = '\'' + class_name + '\':' + class_name + ','

        insert(init_path, 'from .demo_dataset import Demo_Dataset', import_dataset)
        insert(init_path, '     \'Demo_Dataset\' : Demo_Dataset,', import_dataset_all)


def delete_dataset(cfg):
    if not cfg.get('DATA_CONFIG.DATASET'):
        return
    class_name = cfg.DATA_CONFIG.DATASET
    module_dir = Path(cfg.EXP_GROUP_PATH) / Path('pcdet/datasets')
    file_name = str.lower(class_name)
    file_path = Path(module_dir) / Path(file_name + '.py')
    init_path = Path(module_dir) / Path('__init__.py')
    if os.path.exists(file_path):
        os.remove(file_path)
        import_dataset = 'from .' + file_name + ' import ' + class_name
        import_dataset_all = '\'' + class_name + '\':' + class_name + ','
        replace(init_path, import_dataset, '')
        replace(init_path, import_dataset_all, '')


def create_backbone(cfg):
    if not cfg.get('MODEL.BACKBONE.NAME'):
        return
    class_name = cfg.MODEL.BACKBONE.NAME
    module_dir = Path(cfg.EXP_GROUP_PATH) / Path('pcdet/models/backbones')

    file_name = str.lower(class_name)
    file_path = Path(module_dir) / Path(file_name + '.py')
    demo_path = Path(module_dir) / Path('demo_backbone.py')
    init_path = Path(module_dir) / Path('__init__.py')
    if os.path.exists(file_path):
        return
    else:
        shutil.copyfile(demo_path, file_path)
        replace(file_path, 'Demo_Backbone', class_name)
        import_dataset = 'from .' + file_name + ' import ' + class_name
        import_dataset_all = '    \'' + class_name + '\': ' + class_name + ','

        insert(init_path, 'from .demo_backbone import Demo_Backbone', import_dataset)
        insert(init_path, '    \'Demo_Backbone\': Demo_Backbone,', import_dataset_all)


def delete_backbone(cfg):
    if not cfg.get('MODEL.BACKBONE.NAME'):
        return
    class_name = cfg.MODEL.BACKBONE.NAME
    module_dir = Path(cfg.EXP_GROUP_PATH) / Path('pcdet/datasets')

    file_name = str.lower(class_name)
    file_path = Path(module_dir) / Path(file_name + '.py')
    init_path = Path(module_dir) / Path('__init__.py')
    if os.path.exists(file_path):
        os.remove(file_path)
        import_dataset = 'from .' + file_name + ' import ' + class_name
        import_dataset_all = '\'' + class_name + '\':' + class_name + ','
        replace(init_path, import_dataset, '')
        replace(init_path, import_dataset_all, '')



if __name__ == '__main__':
    args, cfg = parse_config()
    create_dataset(cfg)
    create_backbone(cfg)

    delete_dataset(cfg)
    delete_backbone(cfg)
    print(0)
