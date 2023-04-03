import os
import torch
import torch.nn as nn
import numpy as np
from .. import backbones, heads

class ModelTemplate(nn.Module):
    def __init__(self, model_cfg, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.batch_size = 1

        self.module_topology = [
            'backbone', 'head'
        ]
        self.visual_iter = 0

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        return model_info_dict['module_list']

    def build_backbone(self, model_info_dict):
        if self.model_cfg.get('BACKBONE', None) is None:
            return None, model_info_dict

        backbone_module = backbones.__all__[self.model_cfg.BACKBONE.NAME](
            model_cfg=self.model_cfg.BACKBONE,
        )
        return backbone_module, model_info_dict

    def build_head(self, model_info_dict):
        if self.model_cfg.get('HEAD', None) is None:
            return None, model_info_dict
        head_module = heads.__all__[self.model_cfg.HEAD.NAME](
            model_cfg=self.model_cfg.HEAD,
        )
        model_info_dict['module_list'].append(head_module)
        return head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def _load_state_dict(self, model_state_disk, *, strict=True, logger=None):
        state_dict = self.state_dict()  # local cache of state_dict
        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in state_dict and state_dict[key].shape != val.shape:
                # with different spconv versions, we need to adapt weight shapes for spconv blocks
                # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

                val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
                if val_native.shape == state_dict[key].shape:
                    val = val_native.contiguous()
                else:
                    assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
                    val_implicit = val.permute(4, 0, 1, 2, 3)  # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
                    if val_implicit.shape == state_dict[key].shape:
                        val = val_implicit.contiguous()

            if key in state_dict and state_dict[key].shape == val.shape:
                update_model_state[key] = val
                logger.info('Update weight %s: %s' % (key, str(val.shape)))

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
        return state_dict, update_model_state

    def load_params_from_file(self, filename, logger, to_cpu=False):
        print(filename)
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False, logger=logger)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        # epoch = checkpoint.get('epoch', -1)
        # it = checkpoint.get('it', 0.0)
        epoch = 0
        it = 0.0

        self._load_state_dict(checkpoint['model_state'], strict=False, logger=logger)

        if optimizer is not None:
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                try:
                    logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                                % (filename, 'CPU' if to_cpu else 'GPU'))
                    optimizer.load_state_dict(checkpoint['optimizer_state'])
                except:
                    assert filename[-4] == '.', filename
                    src_file, ext = filename[:-4], filename[-3:]
                    optimizer_filename = '%s_optim.%s' % (src_file, ext)
                    if os.path.exists(optimizer_filename):
                        optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                        optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])
            else:
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')
        return it, epoch

