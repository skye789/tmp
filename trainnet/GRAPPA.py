import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import fractional_matrix_power
from pcdet.utils.utils import *
class Grappa:
    '''''
    Input:
        fs_kspace: fully sampled kspace, [kx,ky,coil]
        zf_kspace: zero filled kspace every R-1 rows, 
        acs: auto calibration signal. Usually the central several lines of fully sampled kspace 
        R: accelaration factor
        num_cenLine: number of center line for acs
    Output:
        reconstructed kspace
    '''''

    def __init__(self, fs_kspace, R, num_cenLine):
        self.zf_kspace = self.get_zf_kspace(fs_kspace,R)
        self.num_row_zfKspace, self.num_col_zfKspace, _ = self.zf_kspace.shape

        self.acs = self.get_acs(num_cenLine, fs_kdata=fs_kspace)
        self.num_row_acs, self.num_col_acs, self.nc = self.acs.shape

        self.kernel_shape = (4, 3)
        self.num_row_kernel, self.num_col_kernel = self.kernel_shape  # 4×3

        self.block_width = self.num_col_kernel
        self.block_height = (self.num_row_kernel - 1) * R + 1

        self.nb = int((self.num_row_acs - (self.block_height - 1)) * (self.num_col_acs - 2))  # 10=(R+1)*2
        self.nk = self.num_row_kernel * self.num_col_kernel  # kernel size, 12

        self.row_offset = np.arange(R + 1, R * 2)  # row offset from top left of block to target

        inpo_ksp = self.grappa4x3(flag_acs=False, acs=self.acs,R=R )
        self.img_GRAPPA = self.kspace2img(inpo_ksp)

    def get_img(self):
        return self.img_GRAPPA

    def ifft2c(self,kspace):
        image = np.fft.ifft2(kspace)
        img = np.fft.ifftshift(image)
        return img

    def get_acs(self,num_cenLine, fs_kdata):
        nx, _, _ = fs_kdata.shape
        acs = fs_kdata[(nx - num_cenLine) // 2: (nx + num_cenLine) // 2]
        return acs

    def get_zf_kspace(self,fs_kdata,R ):
        """""
        Input:
            R: acceleration factor
            fs_kdata: fully sampled kdata. (nx, ny, nc)
        Output:
            zf_kspace:fill every other line to zero, if R=2 (nx, ny, nc)
        """""
        nx, _, _ = fs_kdata.shape
        zf_kspace = np.zeros_like(fs_kdata)
        for i in range(nx):
            if i % R == 0:  # even
                zf_kspace[i] = fs_kdata[i]
        return zf_kspace

    def kspace2img(self,inpo_ksp):
        """""
        combine kspace in diff coil and reconstruct image using sum_of_square, 2D
        """""
        m_coil_img = np.zeros_like(inpo_ksp)
        for i in range(self.nc):
            m_coil_img[..., i] = self.ifft2c(inpo_ksp[..., i])
        img_sumSquare = np.linalg.norm(m_coil_img, axis=2)
        return img_sumSquare

    def extract(self, acs, R):
        '''''
        from acs extract source and target                
        '''''
        src = np.zeros((R - 1, self.nb, self.nc * self.nk), dtype=np.complex64)
        targ = np.zeros((R - 1, self.nb, self.nc), dtype=np.complex64)

        for i in range(R - 1):
            src_idx = 0
            for col in range(self.num_col_acs - 2):
                for row in range(self.num_row_acs - (self.block_height - 1)):  # for循环所有target点(nb)

                    block = self.acs[row:row + self.block_height, col:col + self.block_width]

                    src[i, src_idx] = block[::R].flatten()
                    targ[i, src_idx] = acs[row + self.row_offset[i], col + 1]

                    src_idx += 1

        return src, targ

    def interp(self, zp_kspace, ws, R):
        interpolated = np.array(zp_kspace)
        num_row_zpk, num_col_zpk, nc = zp_kspace.shape

        for i in range(R - 1):
            src_idx = 0
            for col in range(num_col_zpk - 2):
                for row in range(0, num_row_zpk - (self.block_height - 1), R):  # for循环所有target点(nb)

                    block = zp_kspace[row:row + self.block_height, col:col + self.block_width]
                    src = block[::R].flatten()

                    interpolated[row + self.row_offset[i], col + 1] = np.dot(src, ws[i])

                    src_idx += 1

        return interpolated

    def zero_padding(self, zp_up, zp_down):
        zp_kdata = np.zeros((self.num_row_zfKspace + zp_up + zp_down, self.num_col_zfKspace + 2, self.nc),
                            np.complex64)
        zp_kdata[zp_up:self.num_row_zfKspace + zp_up, 1:self.num_col_zfKspace + 1, :] = self.zf_kspace
        return zp_kdata

    def grappa4x3(self, flag_acs, acs, R):
        # get source and target in acs region, to calculate weight
        ws = np.zeros((R - 1, self.nk * self.nc, self.nc), dtype=complex)
        src, targ = self.extract(acs,R)  # 3维度
        for i in range(R - 1):
            ws[i] = np.dot(np.linalg.pinv(src[i]), targ[i])

        # zero padding
        zp_up = R
        zp_down = R + 1
        zp_kdata = self.zero_padding(zp_up, zp_down)

        # interpolation
        interpolated = self.interp(zp_kdata, ws,R)
        interpolated = interpolated[zp_up:self.num_row_zfKspace + zp_up, 1:self.num_col_zfKspace + 1, :]

        if flag_acs:
            interpolated[int(self.num_row_zfKspace / 2 - self.num_row_acs / 2)
                         :int(self.num_row_zfKspace / 2 + self.num_row_acs / 2)] = self.acs

        return interpolated


def biuld_grappa_img(kspace,acc):
    '''
    Input
        kspace: [nt, coil, nx, ny]
        Grappa class require[nx,ny,nT]
    Output:
        recon_img = [nt,nx,ny]
    '''
    kspace = kspace.transpose(2,3,1,0) #[kx,ky,coil,nT]
    kx,ky,nCoil,nT = np.shape(kspace)
    num_cenLine = 20
    img_grappa = np.zeros((kx,ky,nT),dtype='complex')
    for t in range(nT):
        grappa = Grappa(fs_kspace=kspace[...,t], R=acc, num_cenLine=num_cenLine)
        img_grappa[...,t] = grappa.get_img()
    img_grappa_swap = img_grappa.transpose(2,0,1) # [nt,nx,ny]
    return img_grappa_swap
