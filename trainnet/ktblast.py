import numpy as np
from skimage.filters import threshold_otsu
from pcdet.utils.utils import *
import matplotlib.pyplot as plt
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import scipy.io
import pickle
from tools.visual_utils.visualizer import Animation


def kt2xf(kt, shift=False, time_axis=-1):
    '''k-t space to x-f space.
    Parameters
    ----------
    kt : array_like
        k-t space data.
    shift: bool, optional
        Perform fftshift when Fourier transforming.
    time_axis : int, optional
        Dimension that holds time data.
    Returns
    -------
    xf : array_like
        Corresponding x-f space data.
    '''

    # Do the transformin' (also move time axis to and fro)
    if not shift:
        return np.moveaxis(np.fft.fft(np.fft.ifft2(
            np.moveaxis(kt, time_axis, -1),
            axes=(0, 1)), axis=-1), -1, time_axis)

    return np.moveaxis(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(
        np.fft.ifftshift(np.fft.fft(
            np.moveaxis(kt, time_axis, -1),
            axis=-1), axes=-1),
        axes=(0, 1)), axes=(0, 1)), axes=(0, 1)), -1, time_axis)

def ktblast(kspace, calib, psi=0.01, R=None, time_axis=-1):
    '''Cartesian k-t BLAST.

    Parameters
    ----------
    kspace : array_like
        Undersampled k-space data.  Nonsampled points should be
        exactly 0.  Assumes a sheared lattice sampling grid.
        [kx,ky,t]
    calib : array_like
        Prior info, usually low-res estimate.
    psi : float, optional
        Noise variance.
    R : int, optional
        Undersampling factor. Estimated from PSF if not provided.
        If this function gets R wrong, bad things will happen.  Check
        to make sure and provide it if necessary.
    time_axis : int, optional
        Dimension that holds time/frequency data.

    Returns
    -------
    recon : array_like
        Reconstructed x-t space.

    Raises
    ------
    AssertionError
        PSF of k-t grid finds more or less than R aliased copies.
        Only raises if R provided.

    Notes
    -----
    Implements the k-t BLAST algorithm as first described in [1]_.
    The Wiener filter expression is given explicitly in [2]_ (see
    equation 1).

    References
    ----------
    .. [1] Tsao, Jeffrey, Peter Boesiger, and Klaas P. Pruessmann.
           "k‐t BLAST and k‐t SENSE: dynamic MRI with high frame rate
           exploiting spatiotemporal correlations." Magnetic
           Resonance in Medicine: An Official Journal of the
           International Society for Magnetic Resonance in Medicine
           50.5 (2003): 1031-1042.
    .. [2] Sigfridsson, Andreas, et al. "Improving temporal fidelity
           in k-t BLAST MRI reconstruction." International Conference
           on Medical Image Computing and Computer-Assisted
           Intervention. Springer, Berlin, Heidelberg, 2007.
    '''

    # Move time axis to end
    kspace = np.moveaxis(kspace, time_axis, -1) #(40,128,128)

    calib = np.moveaxis(calib, time_axis, -1) #(40,128,128) 中间20line采样了

    # Put everything into x-f space
    xf_aliased = kt2xf(kspace, shift=True)
    xf_prior = kt2xf(calib, shift=True)

    # Make sure psi is real (np.cov() can return complex numbers...)
    psi = np.abs(psi)

    # Get sizes
    cx, cy, ct = xf_prior.shape[:]

    # Get PSF of the sampling grid -- don't fftshift because the
    # coordinate system needs to assume center is (0, 0, 0)
    PSF = np.abs(kt2xf(np.abs(kspace) > 0, shift=False))

    # Get indices of locations of aliased copies, should only be R of these
    if R is not None:
        thresh = np.sort(PSF.flatten())[-1*R]  #sort默认升序； 取第R大的x-f域的value
        PSF[PSF < thresh] = 0
        idx = np.where(PSF > 0)  # tuple:3
        '''(0, 64
            0, 0
            0, 20  
                 )'''
        # print('idx:',idx)
        # print(np.stack(idx).shape[1])
        assert np.stack(idx).shape[1] == R, (
            'PSF should define R copies!')
    else:
        thresh = threshold_otsu(PSF)
        idx = np.where(PSF > thresh)
        R = len(idx[0])
        # print('Based on PSF, R is found to be: %d' % R)

    beta = 0.1
    gamma = 2

    # calculate filter (Equation 1 in [2]) -- first get denominator
    axf_prior2 = np.abs(xf_prior)**2
    filter_denom = np.zeros((cx, cy, ct))
    sum_M_alias = 0
    for ii in range(R):
        sum_M_alias += np.roll(
            axf_prior2, (idx[0][ii], idx[1][ii], idx[2][ii]))

    filter_denom = sum_M_alias + psi  #psi is noise variance

    # now divide numerator
    xf_filter = axf_prior2/filter_denom

    # multiply aliased data by filter, rescale, move time axis back
    return np.moveaxis(np.fft.fftshift(np.fft.ifft(np.fft.fftshift(
        xf_aliased*xf_filter*R,
        axes=-1), axis=-1), axes=-1), -1, time_axis)


def build_ktblast_img(x_gnd,acc, ):
    '''
    Undersample in ny direction (phase direction)
    Input
        x_gnd:[nt, nx, ny]
        # mask:[nt,nx,ny]
        # cali_num: number of calibration lines

    Output:
        recon_img = [nt,nx,ny]
    '''
    kt = img2kspace(x_gnd)
    kt = kt.transpose(1,2,0) # [nx,ny,nt]
    cali_num = 10
    nx,ny,nT = kt.shape

    """""training stage"""""
    # crop 40 lines from the center of k-space for calibration
    pd = int(cali_num/2)
    ctr = int(nx / 2)
    kt_calib = np.zeros(kt.shape, dtype=kt.dtype)
    kt_calib[ctr - pd:ctr + pd, ...] = kt[ctr - pd:ctr + pd, ...].copy()
    # kt_calib[:, ctr - pd:ctr + pd, :] = kt[:, ctr - pd:ctr + pd, :].copy()

    # Undersample kspace:
    mask = np.zeros(kt.shape, dtype=bool)
    for i in range(acc):
        mask[:, i::acc, i::acc] = True
    # plt.imshow(mask[:, :, :])
    # plt.show()
    # animation = Animation()
    # animation.show(mask.transpose(2, 0, 1))
    kt *= mask

    # Get noise statistics from non-moving area
    lin = kt2xf(kt, shift=True)[:, 0, 1]   # [nx,ny,nt]
    psi = np.cov(lin)

    # Run k-t BLAST
    recon = ktblast(kt, kt_calib, psi=psi, R=acc)
    recon = recon.transpose(2,0,1) # [nt,nx,ny]
    return recon
