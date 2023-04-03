import torch
import torch.nn as nn
import os

class kt_NEXT_Backbone(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

    def forward(self, batch_dict):

        return batch_dict