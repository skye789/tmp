import numpy as np
from numpy.fft import*
from numpy.linalg import *
import torch
from .dnn_io import *

def combine_coils_in_xf(xf, coil_sens, noiseCov):
    """""
    input xf: [kx,ky,coils,times]
    output xf: [kx,ky,times]
    """""
    nx,ny,coils = np.shape(coil_sens)
    unmix = np.zeros(( nx,ny,coils,1), dtype=complex)
    idx_x, idx_y = np.where(np.sum(coil_sens, axis=2) != 0)  # 零的部分（图像四周）不作处理
    for i in range(len(idx_x)):
        S = np.reshape(coil_sens[idx_x[i], idx_y[i], :], (coils, 1))
        S_H = np.transpose(S.conjugate())
        tmp1 = inv(np.dot(np.dot(S_H, inv(noiseCov)), S))
        tmp2 = np.dot(S_H, inv(noiseCov))
        unmix[idx_x[i], idx_y[i], :,0] = np.dot(tmp1, tmp2)  # unmix= inv( S'*inv(psi)*S ) * S' * inv(psi)
    xf_new = unmix*xf
    xf_mix = np.sum(xf_new, axis=-2)
    return xf_mix

def kspace2img(kspace, dim=(-2,-1)):
    "kt:[times,kx,ky,]"
    "require 2D kspace"
    img = fftshift(ifftn(ifftshift(kspace, axes=dim), axes=dim), axes=dim)
    return img

def img2kspace(img, dim=(-2,-1)):
    "kt:[times,kx,ky,]"
    kspace = ifftshift(fftn(fftshift(img, axes=dim), axes=dim), axes=dim)
    return kspace

def kt2xt(kt, dim=(-2,-1)):
    '''dim is the dimention of nx, ny'''
    xt = kspace2img(kt, dim)
    return xt

def xt2xf(xt, time_dim=0):
    "xt:[times,nx,ny,]"
    xf = ifftshift(fft(fftshift(xt, axes=time_dim), axis= time_dim), axes=time_dim)
    return xf

def xf2xt(xf,time_dim=0):
    "xt:[times,nx,ny,]"
    xt = fftshift(ifft(ifftshift(xf, axes=time_dim), axis= time_dim), axes=time_dim)
    return xt

def estimate_noise_covariance(undersample_xf):
    """""
    input: 
        undersample_xf: [x,y,coils,f]
    return: 
        noise covariance between coils: [num_coils, num_coils]
        
    The calculation of covariance can refer to: https://www.cnblogs.com/geeksongs/p/11190295.html
    noise_covariance计算的是不同coils之间的关系， 我们可以每个coil_img找一个 unmoved single line用于计算 
    """""
    # so choose a single line in measured x-f space that's not moving(the first line)
    # python中covariance计算是基于行向量而不是列向量, noise_data:[nC,X]
    # ; 而matlab基于列向量

    # noise_data = np.reshape(undersample_xf[[0,nx-1],:,:,:], (nC,-1)) * np.sqrt(nx*ny*nT) #[nC, ny]
    noise_data = np.mean(undersample_xf[0,:,:,:], axis=-1)  # noise-only scan (208×15)
    noiseCov = np.cov(np.transpose(noise_data.conjugate()))
    return noiseCov


def kt2xf_lan(kt, time_axis=0, img_axis=(-2,-1)):
    '''''
    input: 
        kt : array_like [times, ..., kx,ky]  
            k-t space data
            ps: low frequency information already in the center
    Returns:   
        xf : array_like [x,y,coils,fre]
            Corresponding x-f space data
    '''''
    xf = fftshift(ifft2(ifftshift(fftshift(fft(ifftshift(kt, axes=time_axis), axis=time_axis), axes=time_axis),
                             axes=img_axis), axes=img_axis), axes=img_axis)
    return xf


# Remove RO oversampling
def remove_oversampling(xt):
    '''x_dim=-2'''
    RO = xt.shape[-2]
    clip_xt = xt[..., RO//4: RO//4*3, :]
    return clip_xt


def complex_psnr(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max
    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).
    '''
    mse = np.mean(np.abs(x - y)**2)
    # mse = np.mean((np.abs(x) - np.abs(y)) ** 2)
    if peak == 'max':
        return 10*np.log10(np.max(np.abs(x))**2/mse)
    else:
        return 10*np.log10(1./mse)

def complex_psnr_torch(x, y, peak='normalized'):
    '''
    x: reference image
    y: reconstructed image
    peak: normalised or max
    Notice that ``abs'' squares
    Be careful with the order, since peak intensity is taken from the reference
    image (taking from reconstruction yields a different value).
    '''
    mse = torch.mean(torch.abs(x - y)**2)
    if peak == 'max':
        return 10*torch.log10(torch.max(torch.abs(x))**2/mse)
    else:
        return 10*torch.log10(1./mse)


def data_dict_torch2np(data_dict):
    for key, val in data_dict.items():
        data_dict[key] = r2c(val.detach().cpu().numpy())
    return data_dict