import os.path

from .model_template import ModelTemplate
from pcdet.utils.dnn_io import *
from pcdet.utils.utils import *

class kt_NEXT_Model(ModelTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict = self.backbone(batch_dict)
        batch_dict = self.head(batch_dict)

        if self.training:
            disp_dict = {}
            visual_dict = self.get_train_visual_dict(batch_dict)
            loss, tb_dict = self.head.get_loss(batch_dict)
            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict, visual_dict
        else:
            disp_dict = self.post_process(batch_dict)
            visual_dict = self.get_test_visual_dict(batch_dict)
            ret_dict = {}
            return ret_dict, disp_dict, visual_dict

    def get_train_visual_dict(self, batch_dict):
        self.visual_iter += 1
        if self.visual_iter % self.dataset.dataset_cfg.SHOW_ITER == 0:
            save_data = {}
            save_data = data_dict_torch2np(save_data)

            visual_data = {'x_recon': batch_dict['x_recon'],
                           }
            visual_data = data_dict_torch2np(visual_data)

            name_data = {'iter': self.visual_iter,
                         'name': os.path.splitext(batch_dict['kt_name'][0])[0],
                         'ACC': self.dataset.dataset_cfg.ACC,
                         'cascades': self.model_cfg.HEAD.cascades,
            }

            visual_dict = {
                'visual_data': visual_data,
                'save_data': save_data,
                'name_data': name_data,
            }
        else:
            visual_dict = None
        return visual_dict

    def get_test_visual_dict(self, batch_dict):
        # b, nt, nx, ny
        save_data = {'xf_recon': batch_dict['xf_recon'],
                     'x_und': batch_dict['x_und'],
                     'xf_gnd': batch_dict['xf_gnd'],
                     'x_gnd': batch_dict['x_gnd'],
                     'x_recon': batch_dict['x_recon'],
                     }
        save_data = data_dict_torch2np(save_data)

        visual_data = {'x_recon': batch_dict['x_recon'],
                       }
        visual_data = data_dict_torch2np(visual_data)

        name_data = {'iter': self.visual_iter,
                     'name': os.path.splitext(batch_dict['kt_name'][0])[0],
                     'ACC': self.dataset.dataset_cfg.ACC,
                     'cascades': self.model_cfg.HEAD.cascades,
                     }

        visual_dict = {
            'visual_data': visual_data,
            'save_data': save_data,
            'name_data': name_data,
        }
        return visual_dict

    def post_process(self, batch_dict):
        psnr = complex_psnr_torch(batch_dict['x_recon'], batch_dict['x_gnd'])
        disp_dict = {'psnr': psnr.item()}
        return disp_dict