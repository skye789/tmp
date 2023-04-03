from collections import namedtuple

import numpy as np
import torch

from .models import build_model

try:
    import kornia
except:
    pass


def build_network(model_cfg, dataset):
    model = build_model(
        model_cfg=model_cfg, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):

    for key, val in batch_dict.items():
        if torch.is_tensor(val):
            batch_dict[key] = val.cuda().float().contiguous()



def model_fn_decorator():
    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict, visual_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict', 'visual_dict'])
        return ModelReturn(loss, tb_dict, disp_dict, visual_dict)
    return model_func
