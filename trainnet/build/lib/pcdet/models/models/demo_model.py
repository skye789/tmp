from .model_template import ModelTemplate

class DemoModel(ModelTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg=model_cfg, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        batch_dict = self.backbone(batch_dict)
        batch_dict = self.head(batch_dict)

        disp_dict = {}
        visual_dict = self.get_visual_dict(batch_dict)
        if self.training:
            loss, tb_dict = self.head.get_loss()
            ret_dict = {'loss': loss}
            return ret_dict, disp_dict, visual_dict, tb_dict
        else:
            ret_dict = self.head.post_processing(batch_dict)
            visual_dict = self.get_visual_dict(batch_dict)
            return ret_dict, disp_dict, visual_dict

    def get_visual_dict(self, batch_dict):
        return batch_dict