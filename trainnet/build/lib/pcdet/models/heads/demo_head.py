import torch
import torch.nn as nn
import torch.nn.functional as F

class Demo_Head(nn.Module):

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def get_loss(self, batch_dict):
        loss = 0
        return loss

    def post_process(self, batch_dict):

        return batch_dict

    def forward(self, batch_dict):
        return batch_dict
