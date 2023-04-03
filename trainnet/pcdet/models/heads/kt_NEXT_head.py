import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
from pcdet.utils.dnn_io import *



def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def xf_CNN(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):
    """
    xf-CNN block in x-f domain
    """

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1

    def conv_i():
        return conv(nf, nf, ks, stride=1, padding=dilation, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)


class CRNN_MRI(nn.Module):
    """
    CRNN-MRI block in image domain
    RNN evolves over temporal dimension only
    """
    def __init__(self, n_ch, nf=64, ks=3, dilation=2):
        super(CRNN_MRI, self).__init__()
        self.nf = nf  #hidden size
        self.ks = ks  #kerme size

        self.bcrnn_1 = BCRNNlayer(n_ch, nf, ks, dilation=1)
        self.bcrnn_2 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_3 = BCRNNlayer(nf, nf, ks, dilation)
        self.bcrnn_4 = BCRNNlayer(nf, nf, ks, dilation)

        self.conv4_x = nn.Conv2d(nf, 2, ks, padding=ks//2)

    def forward(self, x, test=False):

        n_batch, n_ch, width, length, n_seq = x.size()

        x = x.permute(4, 0, 1, 2, 3)  #[30,1,2,256,32]

        out = self.bcrnn_1(x, None, test) ##[30,1,64,256,32]
        out = self.bcrnn_2(out, None, test)
        out = self.bcrnn_3(out, None, test)
        out = self.bcrnn_4(out, None, test)
        out = out.view(-1, self.nf, width, length)
        out = self.conv4_x(out)

        out = out.view(-1, n_batch, 2, width, length)
        out = out.permute(1, 2, 3, 4, 0)

        return out


class kt_NEXT_Head(nn.Module):
    """
    network architecture for k-t NEXT
    """

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.nc = model_cfg.cascades
        self.dilation = model_cfg.dilation
        self.n_channels = model_cfg.n_channels
        self.nd = model_cfg.nd
        self.nf = model_cfg.nf
        self.criterion = torch.nn.MSELoss().cuda()
        xf_conv_blocks = []
        xt_conv_blocks = []
        dcs_xf = []
        dcs_xt = []
        tdxf = []

        for i in range(self.nc):
            xf_conv_blocks.append(xf_CNN(2, self.nd, self.nf, n_out=self.n_channels, conv_dim=2, dilation=self.dilation))
            xt_conv_blocks.append(CRNN_MRI(self.n_channels, self.nf, dilation=self.dilation))
            dcs_xf.append(DataConsistencyInKspace(norm='ortho'))
            dcs_xt.append(DataConsistencyInKspace(norm='ortho'))
            tdxf.append(TransformDataInXfSpaceTA(i > 0, norm=True))

        self.xf_conv_blocks = nn.ModuleList(xf_conv_blocks) # map aliased x_f_diff to unalias xf
        self.xt_conv_blocks = nn.ModuleList(xt_conv_blocks)
        self.dcs_xf = nn.ModuleList(dcs_xf)
        self.dcs_xt = nn.ModuleList(dcs_xt)
        self.tdxf = nn.ModuleList(tdxf)   # save x_f_diff, x_f_avg

    def get_loss(self, data_dict):
        xf_gnd = data_dict['xf_gnd']
        x_gnd = data_dict['x_gnd']
        x_recon = data_dict['x_recon']
        xf_recon = data_dict['xf_recon']
        loss = self.criterion(x_recon, x_gnd) + self.criterion(xf_recon, xf_gnd)
        tb_dict = {'all loss': loss.item()}
        return loss, tb_dict

    def post_process(self, batch_dict):
        ret_dict = {}
        return ret_dict

    def forward(self,data_dict):
        '''require the shape of input is #[batch_size,2,nx,ny,nt]'''
        x = data_dict['x_und']  #[]
        k = data_dict['k_und']
        m = data_dict['mask']

        for i in range(self.nc):
            # x-f domain reconstruction
            xf, xf_avg = self.tdxf[i].perform(x, k, m)
            nb, nc, nx, ny, nt = xf.shape
            xf = xf.permute(0, 3, 1, 2, 4)
            xf = xf.view(-1, nc, nx, nt)  # [ny,2,nx,nt]
            xf_out = self.xf_conv_blocks[i](xf)   # [ny,2,nx,nt]
            xf_out = xf_out.view(-1, ny, 2, nx, nt)
            xf_out = xf_out.permute(0, 2, 3, 1, 4)  # (n, nc, nx, ny, nt)
            xf_recon = xf_out + xf_avg

            # transform signal from x-f domain to image domain
            out_img = fftshift_pytorch(torch_old_ifft(ifftshift_pytorch(xf_out.permute(0, 2, 3, 4, 1), axes=[-2]), 1, normalized=True), axes=[-2])
            out_img = out_img.permute(0, 4, 1, 2, 3)
            x = self.dcs_xf[i].perform(out_img, k, m)

            # image domain reconstruction
            out = self.xt_conv_blocks[i](x)
            x = x + out
            x = self.dcs_xt[i].perform(x, k, m)

        # x [b, 2, nx , ny, nt]
        data_dict['x_recon'] = x
        data_dict['x_gnd'] = data_dict['x_gnd']
        data_dict['xf_recon'] = xf_recon
        data_dict['xf_gnd'] = data_dict['xf_gnd']

        return data_dict

