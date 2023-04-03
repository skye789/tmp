import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class dataConsistencyTerm(nn.Module):

    def __init__(self, noise_lvl=None):
        super(dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            self.noise_lvl = torch.nn.Parameter(torch.from_numpy(np.asarray(noise_lvl)).type(torch.FloatTensor))
            self.noise_lvl.requires_grad = False  # if trainable, set this to true

    def perform(self, x, k0, mask, sensitivity):

        """
        k    - input in k-space
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """
        x = complex_multiply(x[..., 0].unsqueeze(1), x[..., 1].unsqueeze(1),
                             sensitivity[..., 0], sensitivity[..., 1])

        k = torch_old_fft(x, 2, normalized=True)

        v = self.noise_lvl
        if v is not None:  # noisy case
            v = F.relu(self.noise_lvl)  # avoid negative value for noise_lvl
            out = (1 - mask) * k + mask * (v * k + (1 - v) * k0)
        else:  # noiseless case
            out = (1 - mask) * k + mask * k0

        # ### backward op ### #
        x = torch_old_ifft(out, 2, normalized=True)

        Sx = complex_multiply(x[..., 0], x[..., 1],
                              sensitivity[..., 0],
                              -sensitivity[..., 1]).sum(dim=1)
        return Sx


class weightedCouplingTerm(nn.Module):

    def __init__(self, beta=None, gamma=None):
        super(weightedCouplingTerm, self).__init__()
        self.beta = beta
        self.gamma = gamma
        if beta is not None and gamma is not None:
            self.beta = torch.nn.Parameter(torch.from_numpy(np.asarray(beta)).type(torch.FloatTensor))
            self.gamma = torch.nn.Parameter(torch.from_numpy(np.asarray(gamma)).type(torch.FloatTensor))
            self.beta.requires_grad = False  # if trainable, set this to true
            self.gamma.requires_grad = False  # if trainable, set this to true

    def perform(self, im_cnn, xf_cnn, Sx):
        x = self.beta * im_cnn + self.gamma * xf_cnn + (1 - self.beta - self.gamma) * Sx
        return x


class CRNN_MRI_bcrnn_iteration(nn.Module):
    """
    im-CRNN model with 4 BCRNN layers with hidden-to-hidden iteration connections
    """

    def __init__(self, n_ch, nf=64, ks=3, dilation=2, bs=1, iteration=True, **kwargs):
        super(CRNN_MRI_bcrnn_iteration, self).__init__()
        self.nf = nf
        self.ks = ks
        self.bs = bs
        self.iteration = iteration

        self.bcrnn_1 = BCRNNlayer(n_ch, nf, ks, dilation=1, iteration=self.iteration)
        self.bcrnn_2 = BCRNNlayer(nf, nf, ks, dilation, iteration=self.iteration)
        self.bcrnn_3 = BCRNNlayer(nf, nf, ks, dilation, iteration=self.iteration)
        self.bcrnn_4 = BCRNNlayer(nf, nf, ks, dilation, iteration=self.iteration)

        self.conv4_x = nn.Conv2d(nf, 2, ks, padding=ks // 2)

    def forward(self, x, hidden_iteration, test=False):
        n_seq, width, length, n_ch = x.size()
        x = x.view(n_seq // self.bs, self.bs, width, length, n_ch)

        x = x.permute(0, 1, 4, 2, 3)
        hidden_states = []

        out = self.bcrnn_1(x, hidden_iteration[0], test=test)
        hidden_states.append(out)
        out = self.bcrnn_2(out, hidden_iteration[1], test=test)
        hidden_states.append(out)
        out = self.bcrnn_3(out, hidden_iteration[2], test=test)
        hidden_states.append(out)
        out = self.bcrnn_4(out, hidden_iteration[3], test=test)
        hidden_states.append(out)
        out = out.view(-1, self.nf, width, length)
        out = self.conv4_x(out)

        out = out.view(-1, 2, width, length)
        out = out.permute(0, 2, 3, 1)

        return out, hidden_states


class xf_conv_block_iteration(nn.Module):
    """
    xf-CRNN with 4 layers of CRNN-i as denoiser
    """

    def __init__(self, n_ch, nf=64, ks=3, dilation=3, iteration=True, **kwargs):
        super(xf_conv_block_iteration, self).__init__()
        self.nf = nf
        self.ks = ks
        self.iteration = iteration

        self.conv_1 = CRNN_i(n_ch, nf, ks, dilation=1, iteration=self.iteration)
        self.conv_2 = CRNN_i(nf, nf, ks, dilation, iteration=self.iteration)
        self.conv_3 = CRNN_i(nf, nf, ks, dilation, iteration=self.iteration)
        self.conv_4 = CRNN_i(nf, nf, ks, dilation, iteration=self.iteration)

        self.conv_5 = nn.Conv2d(nf, 2, ks, padding=ks // 2)

    def forward(self, x, hidden_iteration):
        hidden_states = []
        out = self.conv_1(x, hidden_iteration[0])
        hidden_states.append(out)
        out = self.conv_2(out, hidden_iteration[1])
        hidden_states.append(out)
        out = self.conv_3(out, hidden_iteration[2])
        hidden_states.append(out)
        out = self.conv_4(out, hidden_iteration[3])
        hidden_states.append(out)

        out = self.conv_5(out)

        return out, hidden_states

class CTF_Head(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.cascades = model_cfg.cascades

        self.conv_blocks = CRNN_MRI_bcrnn_iteration(n_ch=2, nf=64, dilation=3, iteration=True)
        self.xf_conv_blocks = xf_conv_block_iteration(n_ch=2, nf=64, dilation=3, iteration=True)
        self.pdc_blocks = dataConsistencyTerm(model_cfg.alfa)
        self.wcp_blocks = weightedCouplingTerm(model_cfg.beta, model_cfg.gamma)
        self.tdxf = TransformDataInXfSpaceTA_mc(False, norm=True)
        self.tdxt = TransformDataInXtSpaceTA_mc(False, norm=True)

        self.criterion = nn.L1Loss().cuda()


    def get_loss(self, batch_dict):
        x_recon = batch_dict['x_recon']
        x_gnd = batch_dict['x_gnd']
        loss = self.criterion(x_recon, x_gnd)
        tb_dict = {'all loss': loss.item()}
        return loss, tb_dict

    def post_process(self, batch_dict):
        ret_dict = {}
        return ret_dict

    def forward(self, data_dict):
        x_und = data_dict['x_und'].permute(2, 0, 3, 4, 1)    # [nt, b, nx, ny, 2]
        k_und = data_dict['k_und'].permute(2, 0, 3, 4, 5, 1)   # [nt, b, nCoil, nx, ny, 2]
        mask = data_dict['mask'].permute(2, 0, 3, 4, 5, 1)   # [nt, b, nCoil, nx, ny, 2]
        coil_sens = data_dict['coil_sens'].permute(2, 0, 3, 4, 5, 1)  # [nt, b, nCoil, nx, ny, 2]
        x_gnd = data_dict['x_gnd'].permute(2, 0, 3, 4, 1)


        nt, batch_size, n_coil, nx, ny, n_ch = k_und.size()
        size_h = [4, nt, batch_size, 64, nx, ny]

        if self.training:
            hidden_im = torch.zeros(size_h).to(x_und.device)
            hidden_xf = torch.zeros([4, nx, 64, ny, nt]).to(x_und.device)
        else:
            with torch.no_grad():
                hidden_im = torch.zeros(size_h).to(x_und.device)
                hidden_xf = torch.zeros([4, nx, 64, ny, nt]).to(x_und.device)

        k = k_und.view(-1, n_coil, nx, ny, n_ch)
        m = mask.view(-1, n_coil, nx, ny, n_ch)
        c = coil_sens.view(-1, n_coil, nx, ny, n_ch)
        x = x_und.view(-1, nx, ny, n_ch)

        x_avg = self.tdxt.perform(x, k, m, c)
        for i in range(self.cascades):
            # x-f space reconstruction
            xf, xf_avg = self.tdxf.perform(x, k, m, c)
            xf = xf.permute(1, 3, 2, 0)  # [nx, nc, ny, nt]
            xf_out, hidden_xf = self.xf_conv_blocks(xf, hidden_xf)
            xf_out = xf_out.permute(3, 0, 2, 1)  # (nt, nx, ny, nc)
            xf_out = xf_out + xf_avg
            out_img = fftshift_pytorch(
                torch_old_ifft(ifftshift_pytorch(xf_out.permute(1, 2, 0, 3), axes=[-2]), 1, normalized=True), axes=[-2])
            out_img = out_img.permute(2, 0, 1, 3)

            # image space reconstruction
            x_cnn, hidden_im = self.conv_blocks(torch.sub(x, x_avg), hidden_im)
            x_cnn = x_avg + x_cnn
            # data consistency
            Sx = self.pdc_blocks.perform(x, k, m, c)
            # weighted combination
            x_recon = self.wcp_blocks.perform(x_cnn, out_img, Sx)
        x_recon = x_recon.reshape(nt, batch_size, nx, ny, n_ch)  # (nt, batch_size=1, nx, ny, n_ch=2)

        data_dict['x_und'] = x_und
        data_dict['k_und'] = k_und
        data_dict['mask'] = mask
        data_dict['coil_sens'] = coil_sens
        data_dict['x_recon'] = x_recon
        data_dict['x_gnd'] = x_gnd

        return data_dict



