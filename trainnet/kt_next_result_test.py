import numpy as np
from pcdet.utils.utils import complex_psnr
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_root_mse
from skimage.metrics import peak_signal_noise_ratio
import pcdet.utils.compressed_sensing as cs
from tools.visual_utils.visualizer import Animation
from pcdet.utils.utils import *
import pickle
import matplotlib.pyplot as plt
from GRAPPA import biuld_grappa_img
from ktblast import build_ktblast_img
import pygrappa
from gro import GRO
import h5py

def map(image):
    image = np.abs(image)
    max = np.max(image)
    min = np.min(image)
    return (image-min)/max-min

def display(title, x_gnd, recon_img):
    print(title)
    psnr = peak_signal_noise_ratio(np.abs(x_gnd),np.abs(recon_img),data_range=np.abs(np.max(x_gnd)))
    print('psnr: ', "%.3f" % psnr)
    ssim = structural_similarity(np.abs(x_gnd),np.abs(recon_img),data_range=np.abs(np.max(x_gnd)))
    print('ssim: ', "%.3f" %ssim)
    NRMSE = normalized_root_mse(np.abs(x_gnd),np.abs(recon_img))
    print('NRMSE', "%.3f" %NRMSE)
    print()

def ifft2c(kspace):
    image = np.fft.ifft2(kspace)
    # image = np.fft.ifftshift(image)
    return image

def build_gnd_xt(data_path):
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
    nt, nC, nx, ny = data_dict['kt'].shape
    data_dict['coil_sens'] = np.tile(data_dict['coil_sens'][np.newaxis], (nt, 1, 1, 1))  # [nt, coil, nx, ny]
    scale = 10000
    kt = data_dict['kt'] * scale  # [nt, coil, nx, ny] # 16, 20, 192, 144
    xt = kt2xt(kt)
    coil_sens = data_dict['coil_sens']
    x_gnd = np.sum(coil_sens.conjugate() * xt, axis=1) / (np.linalg.norm(coil_sens, axis=1) + 1e-8)  # [nt,nx,ny]
    return x_gnd, kt

def show_time_series(x_recon):
    nt = x_recon.shape[0]
    for i in range(nt):
        plt.subplot(2,8,i+1)
        plt.imshow(np.abs(x_recon[i]), cmap='gray')
        plt.axis('off')
    plt.show()

def get_mask(kt, ACC, pattern='uniform'):
    '''
    Undersample in ny direction (phase direction,  →)
    Input
        kt:[nt, nC, nx, ny]

    Output:
        mask, k_und, x_und:[nt,nC,nx,ny]
    '''
    print("ACC,", ACC)
    print("sample pattern", pattern)
    nt, nC, nx, ny = kt.shape
    cali_num = None
    if pattern=='GRAPPA' or pattern=='ktBLAST' :
        '''
        uniform undersample mask,
        but first consider the calibration lines into the total sample lines
        mask doesn't vary along time dimension
        '''
        mask = np.zeros_like(kt)
        cali_num = 8
        mask[:,:,:,(ny-cali_num)//2:(ny+cali_num)//2] = True
        undersam_num = ny/ACC-cali_num
        interval = int((ny-cali_num)//undersam_num)
        print("interval", interval)
        mask[..., (ny // 2 - cali_num // 2)::-interval] = True
        mask[..., (ny // 2 + cali_num // 2 )::interval] = True

    elif pattern=='GRO':
        '''
        GRO pattern (GRO = golden ratio offset)
        random undersample mask, sample more in the center
        '''
        mask = GRO(ny // ACC, nt, ny)  # [144,16,1]
        mask = np.squeeze(mask)
        mask = np.expand_dims(mask, 0).repeat(nx, 0)
        mask = np.expand_dims(mask, 0).repeat(nC, 0)
        mask = mask.transpose(3, 0, 1, 2)  # 16, 20, 192, 144
        # print('size of mask_gro', np.shape(mask))
    else:  #uniform undersample mask,mask vary along with time diemention
        mask = np.zeros_like(kt)
        for i in range(ACC):
            mask[i::ACC, :, :, i::ACC] = True

    k_und = mask * kt  # 16, 20, 192, 144
    x_und = kt2xt(k_und)

    # plt.subplot(1, 2, 1)
    # plt.imshow(np.abs(mask[10, 3]), cmap='gray')
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.abs(mask[11, 3]), cmap='gray')
    # plt.axis('off')
    # plt.show()

    return mask, k_und, x_und, cali_num

def show_all_mask(kt):
    Accelerations = [4, 8, 12]
    for i, acc in enumerate(Accelerations):
        names = ["{}×unifrom mask with ACS".format(acc),"{}×unifrom mask ".format(acc), "{}×GRO mask".format(acc)]
        mask_GRAPPA, k_und, x_und, cali_num = get_mask(kt, ACC=acc, pattern='GRAPPA')
        mask_ktBLAST, k_und, x_und, cali_num = get_mask(kt, ACC=acc, pattern='ktBLAST')
        mask_GRO, k_und, x_und, cali_num = get_mask(kt, ACC=acc, pattern='GRO')

        plt.subplot(3, 3, i * 3 + 1)
        plt.imshow(np.abs(mask_GRAPPA[10, 0]), cmap='gray')
        plt.title(names[0])
        plt.axis('off')
        plt.subplot(3, 3, i * 3 + 2)
        plt.imshow(np.abs(mask_ktBLAST[10, 0]), cmap='gray')
        plt.title(names[1])
        plt.axis('off')
        plt.subplot(3, 3, i * 3 + 3)
        plt.imshow(np.abs(mask_GRO[10, 0]), cmap='gray')
        plt.title(names[2])
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    '''create groundtruth'''
    raw_data_path = 'raw_data/fs_0038_3T.pkl'
    x_gnd,kt = build_gnd_xt(raw_data_path)  #[nt,nx,ny]   # kt: [nt, coil, nx, ny]
    x_gnd = map(x_gnd)
    nt, nC, nx, ny = kt.shape

    method = "ktBLAST"   # GRAPPA  ktBLAST  CS   ktNEXT  VarNet

    '''compare the result of GRAPPA'''
    if method == "GRAPPA":
        # Accelerations = [4,8,12]
        Accelerations = [8]
        animation = Animation(output_dir='output/video')
        for i,acc in enumerate(Accelerations):
            mask, k_und, x_und, cali_num = get_mask(kt, ACC=acc, pattern='GRAPPA')
            x_und_nocoil = np.linalg.norm(x_und,axis=1)
            names = ["{}×unifrom mask with ACS".format(acc), "{}×undersample".format(acc), "{}×recon".format(acc),"{}×error".format(acc)]
            title = 'recon_img_grappa,acc={}'.format(acc)
            recon_img_grappa = np.zeros((nt, nx, ny), dtype='complex')

            for t in range(nt):
                calib = k_und[t, :, :,(ny-cali_num)//2:(ny+cali_num)//2]
                recon_kt = pygrappa.grappa(kspace=k_und[t], calib=calib,coil_axis=0)
                recon_img = kt2xt(recon_kt)
                recon_img_grappa[t] = np.linalg.norm(recon_img,axis=0)
            recon_img_grappa = map(recon_img_grappa)

            # display(title, x_gnd, recon_img_grappa)
            #     # animation.show(recon_img_grappa,title=title)
            # animation.save_data_pkl(recon_img_grappa, title=title)
            animation.video_builder(recon_img_grappa, title=title)

            # plt.subplot(3, 5, i * 5 + 1)
            # plt.imshow(np.abs(mask[10, 0]), cmap='gray')
            # plt.title(names[0])
            # plt.axis('off')
            # plt.subplot(3, 5, i * 5 + 2)
            # plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
            # plt.title(names[1])
            # plt.axis('off')
            # plt.subplot(3, 5, i * 5 + 3)
            # plt.imshow(np.abs(recon_img_grappa[10]), cmap='gray')
            # plt.title(names[2])
            # plt.axis('off')
            # plt.subplot(3, 5, i * 5 + 4)
            # plt.imshow(np.abs(recon_img_grappa[10] - x_gnd[10]), cmap='gray')
            # plt.title(names[3])
            # plt.axis('off')
            # display(title, x_gnd, recon_img_grappa)
            # plt.subplot(3, 5, i * 5 + 5)
            # plt.imshow(np.abs(recon_img_grappa[:,:,96].transpose(1,0)), cmap='gray')
            # plt.title(names[4])
            # plt.axis('off')

            names = ["unifrom mask with ACS", "undersample", "recon", "error", "xt"]
            xt = np.ones((nx,ny))
            xt[:,:nt]= recon_img_grappa[:, :, 96].transpose(1, 0)
            plt.subplot(1, 5, 1)
            plt.imshow(np.abs(mask[10, 0]), cmap='gray')
            plt.title(names[0])
            plt.axis('off')
            plt.subplot(1, 5, 2)
            plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
            plt.title(names[1])
            plt.axis('off')
            plt.subplot(1, 5, 3)
            plt.imshow(np.abs(recon_img_grappa[10]), cmap='gray')
            plt.title(names[2])
            plt.axis('off')
            plt.subplot(1, 5, 4)
            plt.imshow(np.abs(recon_img_grappa[10] - x_gnd[10]), cmap='gray')
            plt.title(names[3])
            plt.axis('off')
            plt.subplot(1, 5, 5)
            plt.imshow(np.abs(xt), cmap='gray')
            plt.title(names[4], loc='left')
            plt.axis('off')
        plt.show()

    '''compare the result of ktBLAST'''
    if method == "ktBLAST":
        # Accelerations = [4,8,12]
        Accelerations = [8]
        animation = Animation(output_dir='output/video')
        for i,acc in enumerate(Accelerations):
            mask, k_und, x_und, cali_num = get_mask(kt, ACC=acc, pattern='ktBLAST')
            x_und_nocoil = np.linalg.norm(x_und, axis=1)
            names = ["{}×uniform mask".format(acc),"{}×undersample".format(acc), "{}×recon".format(acc), "{}×error".format(acc)]
            title = 'recon_img_ktblast,acc={}'.format(acc)
            recon_img_ktblast = build_ktblast_img(x_gnd.transpose(0,2,1),acc)
            recon_img_ktblast = recon_img_ktblast.transpose(0,2,1)
            recon_img_ktblast = map(recon_img_ktblast)

            # animation.show(recon_img_ktblast,title=title)
            # animation.save_data_pkl(recon_img_ktblast, title=title)
            # animation.video_builder(recon_img_ktblast, title=title)
            # show_time_series(recon_img_ktblast)

            # plt.subplot(3, 4, i * 4 + 1)
            # plt.imshow(np.abs(mask[10, 0]), cmap='gray')
            # plt.title(names[0])
            # plt.axis('off')
            # plt.subplot(3, 4, i * 4 + 2)
            # plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
            # plt.title(names[1])
            # plt.axis('off')
            # plt.subplot(3, 4, i * 4 + 3)
            # plt.imshow(np.abs(recon_img_ktblast[10]), cmap='gray')
            # plt.title(names[2])
            # plt.axis('off')
            # plt.subplot(3, 4, i * 4 + 4)
            # plt.imshow(np.abs(recon_img_ktblast[10] - x_gnd[10]), cmap='gray')
            # plt.title(names[3])
            # plt.axis('off')
            # display(title, x_gnd, recon_img_ktblast)

            names = ["ktBLAST mask", "undersample", "recon", "error", "xt"]
            xt = np.ones((nx, ny))
            xt[:, :nt] = recon_img_ktblast[:, :, 96].transpose(1, 0)
            plt.subplot(1, 5, 1)
            plt.imshow(np.abs(mask[10, 0]), cmap='gray')
            plt.title(names[0])
            plt.axis('off')
            plt.subplot(1, 5, 2)
            plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
            plt.title(names[1])
            plt.axis('off')
            plt.subplot(1, 5, 3)
            plt.imshow(np.abs(recon_img_ktblast[10]), cmap='gray')
            plt.title(names[2])
            plt.axis('off')
            plt.subplot(1, 5, 4)
            plt.imshow(np.abs(recon_img_ktblast[10] - x_gnd[10]), cmap='gray')
            plt.title(names[3])
            plt.axis('off')
            plt.subplot(1, 5, 5)
            plt.imshow(np.abs(xt), cmap='gray')
            plt.title(names[4], loc='left')
            plt.axis('off')
        plt.show()

    '''compare the result of compress sensing'''
    if method == "CS":
        animation = Animation(output_dir='output/video')
        restrict = "FFT"  #TV  FFT
        Accelerations = [4,8,12]
        # Accelerations = [8]
        for i,acc in enumerate(Accelerations):
            names = ["{}×GRO mask".format(acc),"{}×undersample".format(acc), "{}×recon".format(acc), "{}×error".format(acc)]
            mask, k_und, x_und, cali_num = get_mask(kt, ACC=acc, pattern='GRO')
            x_und_nocoil = np.linalg.norm(x_und, axis=1)
            nT, coil, kx, ky = np.shape(kt)
            if restrict == "TV":
                title = 'recon_img_CS_TV,acc={}'.format(acc)
                path = 'output/cs_total_variation/cs_tv,ACC{}.pkl'.format(acc)
            else :
                title = 'recon_img_CS_FFT,acc={}'.format(acc)
                path = 'output/cs_fft/cs_fft,ACC{}.pkl'.format(acc)
            with open(path, 'rb') as f:
                recon_img_CS = pickle.load(f)
            recon_img_CS = map(recon_img_CS)
            display(title, x_gnd, recon_img_CS)
            # show_time_series(recon_img_CS)

            # display(title, x_gnd, recon_img_grappa)
            #     # animation.show(recon_img_grappa,title=title)
            #     # animation.save_data_pkl(recon_img_grappa, title=title)
            animation.video_builder(recon_img_CS, title=title)

            # plt.subplot(3, 4, i * 4 + 1)
            # plt.imshow(np.abs(mask[10,0]), cmap='gray')
            # plt.title(names[0])
            # plt.axis('off')
            # plt.subplot(3, 4, i * 4 + 2)
            # plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
            # plt.title(names[1])
            # plt.axis('off')
            # plt.subplot(3, 4, i * 4 + 3)
            # plt.imshow(np.abs(recon_img_CS[10]), cmap='gray')
            # plt.title(names[2])
            # plt.axis('off')
            # plt.subplot(3, 4, i * 4 + 4)
            # plt.imshow(np.abs(recon_img_CS[10] - x_gnd[10]), cmap='gray')
            # plt.title(names[3])
            # plt.axis('off')


            # names = ["GRO mask", "undersample", "recon", "error", "xt"]
            # xt = np.ones((nx, ny))
            # xt[:, :nt] = recon_img_CS[:, :, 96].transpose(1, 0)
            # plt.subplot(1, 5, 1)
            # plt.imshow(np.abs(mask[10, 0]), cmap='gray')
            # plt.title(names[0])
            # plt.axis('off')
            # plt.subplot(1, 5, 2)
            # plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
            # plt.title(names[1])
            # plt.axis('off')
            # plt.subplot(1, 5, 3)
            # plt.imshow(np.abs(recon_img_CS[10]), cmap='gray')
            # plt.title(names[2])
            # plt.axis('off')
            # plt.subplot(1, 5, 4)
            # plt.imshow(np.abs(recon_img_CS[10] - x_gnd[10]), cmap='gray')
            # plt.title(names[3])
            # plt.axis('off')
            # plt.subplot(1, 5, 5)
            # plt.imshow(np.abs(xt), cmap='gray')
            # plt.title(names[4], loc='left')
            # plt.axis('off')
        # plt.show()

    '''compare the result of ktNext'''
    if method == "ktNEXT":
        # AccCascades = [(4, 1), (8, 1), (12,1), (4, 2), (8, 2), (12, 2), (4, 3), (8, 3), (12, 3), (4,4), (8, 4), (12, 4)]
        # AccCascades = [ (4, 2), (8, 2), (12, 2)]
        # AccCascades = [(8, 1),(8, 2), (8, 3), (8, 4)]
        AccCascades = [(8,2)]

        for i,data in enumerate(AccCascades):
            animation = Animation(output_dir='output/video')
            acc = data[0]
            cas = data[1]
            path ="E:/MRI_result/CAS{}ACC{}/visual/eval_train/iter_780_name_fs_0038_3T_ACC_{}_cascades_{}_.pkl".format(cas,acc,acc,cas)
            with open(path, 'rb') as f:
                kt_next_recon_img = pickle.load(f)
            kt_next_recon_img = map(kt_next_recon_img)
            title = 'acc{}; cas{}'.format(acc, cas)
            names = ["{}×unform mask".format(acc), "{}×undersample".format(acc), "{}×recon".format(acc), "{}×error".format(acc)]
            cas_names = ["{}cascade,recon".format(cas), "{}cascade,error".format(cas)]
            # show_time_series(kt_next_recon_img)

            mask = np.zeros_like(x_gnd)  #[nt,nx,ny]
            for i in range(acc):
                mask[i::acc, :, i::acc] = True
            k_und_nocoil = img2kspace(x_gnd) * mask
            x_und_nocoil = kt2xt(k_und_nocoil)

            k_avg = np.mean(k_und_nocoil,axis=0)
            x_avg = kt2xt(k_avg)

            # for t in range(12):
            #     # tmp = kt2xt(k_und_nocoil[t]-k_avg)
            #     plt.subplot(3,4,1+t)
            #     plt.imshow(np.abs(kt_next_recon_img[:,70+t]), cmap='gray')
            #     # plt.title("t={}".format(t+1))
            # #     plt.imshow(np.abs(x_und_nocoil[t]), cmap='gray')
            #     plt.axis('off')
            # plt.show()

            # k_diff = k_avg-k_und_nocoil
            # k_diff[:,:71] = img2kspace(x_gnd)[:,:71]
            # img = kt2xt(k_diff)
            # xt = kt_next_recon_img[:,:,96]  #第96列
            # plt.imshow(np.abs(img[:,:,96]), cmap='gray')
            # plt.imshow(np.log(np.abs(k_diff[0])), cmap='gray')
            # plt.axis('off')
            # plt.show()

            # display(title, x_gnd, kt_next_recon_img)
            #     # animation.show(recon_img_grappa,title=title)
            #     # animation.save_data_pkl(recon_img_grappa, title=title)
            animation.video_builder(kt_next_recon_img, title=title)

            # plt.subplot(3, 4, i * 4 + 1)
            # plt.imshow(np.abs(mask[10]), cmap='gray')
            # plt.title(names[0])
            # plt.axis('off')
            # plt.subplot(3, 4, i * 4 + 2)
            # plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
            # plt.title(names[1])
            # plt.axis('off')
            # plt.subplot(3, 4, i * 4 + 3)
            # plt.imshow(np.abs(kt_next_recon_img[10]), cmap='gray')
            # plt.title(names[2])
            # plt.axis('off')
            # plt.subplot(3, 4, i * 4 + 4)
            # plt.imshow(np.abs(kt_next_recon_img[10] - x_gnd[10]), cmap='gray')
            # plt.title(names[3])
            # plt.axis('off')

        #     plt.subplot(2, 5, i + 2)
        #     plt.imshow(np.abs(kt_next_recon_img[10]), cmap='gray')
        #     plt.title(cas_names[0])
        #     plt.axis('off')
        #     plt.subplot(2, 5, 5 + i +2)
        #     plt.imshow(np.abs(kt_next_recon_img[10] - x_gnd[10]), cmap='gray')
        #     plt.title(cas_names[1])
        #     plt.axis('off')
        # plt.subplot(2, 5, 1)
        # plt.imshow(np.abs(x_gnd[10]), cmap='gray')
        # plt.title("ground truth")
        # plt.axis('off')
        # plt.subplot(2, 5, 6)
        # plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
        # plt.title("8×undersample")
        # plt.axis('off')

            # names = ["uniform mask", "undersample", "recon", "error", "xt"]
            #
            # xt = np.ones((nx, ny))
            # xt[:, :nt] = kt_next_recon_img[:, :, 96].transpose(1, 0)
            #
            # plt.subplot(1, 5, 1)
            # plt.imshow(np.abs(mask[10]), cmap='gray')
            # plt.title(names[0])
            # plt.axis('off')
            # plt.subplot(1, 5, 2)
            # plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
            # plt.title(names[1])
            # plt.axis('off')
            # plt.subplot(1, 5, 3)
            # plt.imshow(np.abs(kt_next_recon_img[10]), cmap='gray')
            # plt.title(names[2])
            # plt.axis('off')
            # plt.subplot(1, 5, 4)
            # plt.imshow(np.abs(kt_next_recon_img[10] - x_gnd[10]), cmap='gray')
            # plt.title(names[3])
            # plt.axis('off')
            # plt.subplot(1, 5, 5)
            # plt.imshow(np.abs(xt), cmap='gray')
            # plt.title(names[4], loc='left')
            # plt.axis('off')

        # plt.show()

    '''compare the result of VarNet'''
    if method == "VarNet":
        # Accelerations = [4, 8, 12]
        Accelerations = [8]
        for i, acc in enumerate(Accelerations):
            animation = Animation(output_dir='output/video')
            names = ["{}×GRO mask".format(acc), "{}×undersample".format(acc), "{}×recon".format(acc), "{}×error".format(acc)]
            mask, k_und, x_und, cali_num = get_mask(kt, ACC=acc, pattern='GRO')
            x_und_nocoil = np.linalg.norm(x_und, axis=1)
            nT, coil, kx, ky = np.shape(kt)

            title = 'recon_img_VarNet,acc={}'.format(acc)
            path = 'output/VarNet/results/CineVN_nan_{}x/reconstructions/ocmr_test_gro_{}/fs_0038_3T_slice00.h5'.format(str(acc).zfill(2),str(acc).zfill(2))
            # path = 'output/VarNet/results/CineVN_nan_small_{}x/reconstructions/ocmr_test_gro_{}/fs_0038_3T_slice00.h5'.format(str(acc).zfill(2),str(acc).zfill(2))

            recon_img_var = h5py.File(path, 'r')  # 打开h5文件
            recon_img_var = np.squeeze(recon_img_var["reconstruction"])
            recon_img_var = np.flip(recon_img_var, axis=(-2,-1))
            recon_img_var = map(recon_img_var)

            # show_time_series(recon_img_var)
            display(title, x_gnd, recon_img_var)
            #     # animation.show(recon_img_grappa,title=title)
            #     # animation.save_data_pkl(recon_img_grappa, title=title)
            animation.video_builder(recon_img_var, title=title)

        #     plt.subplot(3, 4, i * 4 + 1)
        #     plt.imshow(np.abs(mask[10,0]), cmap='gray')
        #     plt.title(names[0])
        #     plt.axis('off')
        #     plt.subplot(3, 4, i * 4 + 2)
        #     plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
        #     plt.title(names[1])
        #     plt.axis('off')
        #     plt.subplot(3, 4, i * 4 + 3)
        #     plt.imshow(np.abs(recon_img_var[10]), cmap='gray')
        #     plt.title(names[2])
        #     plt.axis('off')
        #     plt.subplot(3, 4, i * 4 + 4)
        #     plt.imshow(np.abs(recon_img_var[10] - x_gnd[10]), cmap='gray')
        #     plt.title(names[3])
        #     plt.axis('off')

            names = ["GRO mask", "undersample", "recon", "error", "xt"]
            xt = np.ones((nx, ny))
            xt[:, :nt] = recon_img_var[:, :, 96].transpose(1, 0)

            # plt.subplot(1, 5, 1)
            # plt.imshow(np.abs(mask[10,0]), cmap='gray')
            # plt.title(names[0])
            # plt.axis('off')
            # plt.subplot(1, 5, 2)
            # plt.imshow(np.abs(x_und_nocoil[10]), cmap='gray')
            # plt.title(names[1])
            # plt.axis('off')
            # plt.subplot(1, 5, 3)
            # plt.imshow(np.abs(recon_img_var[10]), cmap='gray')
            # plt.title(names[2])
            # plt.axis('off')
            # plt.subplot(1, 5, 4)
            # plt.imshow(np.abs(recon_img_var[10] - x_gnd[10]), cmap='gray')
            # plt.title(names[3])
            # plt.axis('off')
            # plt.subplot(1, 5, 5)
            # plt.imshow(np.abs(xt), cmap='gray')
            # plt.title(names[4], loc='left')
            # plt.axis('off')

            '''ground truth'''
            mask = np.zeros((nx,ny))
            xt = np.ones((nx, ny))
            xt[:, :nt] = x_gnd[:, :, 96].transpose(1, 0)
            plt.subplot(1, 5, 1)
            plt.imshow(np.abs(mask), cmap='gray')
            plt.title(names[0])
            plt.axis('off')
            plt.subplot(1, 5, 2)
            plt.imshow(np.abs(x_gnd[10]), cmap='gray')
            plt.title(names[1])
            plt.axis('off')
            plt.subplot(1, 5, 3)
            plt.imshow(np.abs(x_gnd[10]), cmap='gray')
            plt.title(names[2])
            plt.axis('off')
            plt.subplot(1, 5, 4)
            plt.imshow(np.abs(x_gnd[10] - x_gnd[10]), cmap='gray')
            plt.title(names[3])
            plt.axis('off')
            plt.subplot(1, 5, 5)
            plt.imshow(np.abs(xt), cmap='gray')
            plt.title(names[4], loc='left')
            plt.axis('off')

        plt.show()



    # '''compare different method'''
    # visualError = False
    # AccCascades = [(8, 1), (8, 2), (8, 3), (8, 4)]
    # ACC = 8
    # mask = cs.shear_grid_mask(x_gnd.shape, ACC, sample_low_freq=True, sample_n=4)
    # x_und, k_und = cs.undersample(x_gnd, mask, centred=False, norm='ortho')  # [nt,nx.ny]
    # data = AccCascades[0]
    # path = 'output/kt-ACC-{}-cascades-{}-scale-10000/visual/eval/iter_1300_name_fs_0038_3T_ACC_{}_cascades_{}_.pkl'.format(
    #     data[0], data[1], data[0], data[1])
    # with open(path, 'rb') as f:
    #     kt_next_recon_img0 = pickle.load(f)
    # data = AccCascades[1]
    # path = 'output/kt-ACC-{}-cascades-{}-scale-10000/visual/eval/iter_1300_name_fs_0038_3T_ACC_{}_cascades_{}_.pkl'.format(
    #     data[0], data[1], data[0], data[1])
    # with open(path, 'rb') as f:
    #     kt_next_recon_img1 = pickle.load(f)
    # data = AccCascades[2]
    # path = 'output/kt-ACC-{}-cascades-{}-scale-10000/visual/eval/iter_1300_name_fs_0038_3T_ACC_{}_cascades_{}_.pkl'.format(
    #     data[0], data[1], data[0], data[1])
    # with open(path, 'rb') as f:
    #     kt_next_recon_img2 = pickle.load(f)
    # data = AccCascades[3]
    # path = 'output/kt-ACC-{}-cascades-{}-scale-10000/visual/eval/iter_1300_name_fs_0038_3T_ACC_{}_cascades_{}_.pkl'.format(
    #     data[0], data[1], data[0], data[1])
    # with open(path, 'rb') as f:
    #     kt_next_recon_img3 = pickle.load(f)
    #
    # path = 'output/cs_total_variation/compress_sense,ACC{}.pkl'.format(ACC)
    # with open(path, 'rb') as f:
    #     recon_img_CS8 = pickle.load(f)
    #
    #
    #
    # fontsize = 12
    # plt.subplot(2, 4, 1)
    # plt.title("({}) Ground Truth".format(chr(ord("a"))), fontsize=fontsize)
    # if not  visualError:
    #     plt.imshow(np.abs(x_gnd[10]-x_gnd[10]), cmap='gray')
    # else:
    #     plt.imshow(np.abs(x_gnd[10]), cmap='gray')
    # plt.axis('off')
    # plt.subplot(2, 4, 2)
    # plt.title("({}) GRAPPA".format(chr(ord("b"))), fontsize=fontsize)
    # if not visualError:
    #     plt.imshow(np.abs(recon_img_grappa8[10] - x_gnd[10]), cmap='gray')
    # else:
    #     plt.imshow(np.abs(recon_img_grappa8[10]), cmap='gray')
    # plt.axis('off')
    # plt.subplot(2, 4, 3)
    # plt.title("({}) Improved kt-BLAST".format(chr(ord("c"))), fontsize=fontsize)
    # if not visualError:
    #     plt.imshow(np.abs(recon_img_ktblast8[10] - x_gnd[10]), cmap='gray')
    # else:
    #     plt.imshow(np.abs(recon_img_ktblast8[10]), cmap='gray')
    # plt.axis('off')
    #
    # plt.subplot(2, 4, 4)
    # plt.title("({}) CS+TV".format(chr(ord("d"))), fontsize=fontsize)
    # if not visualError:
    #     plt.imshow(np.abs(recon_img_CS8[10] - x_gnd[10]), cmap='gray')
    # else:
    #     plt.imshow(np.abs(recon_img_CS8[10]), cmap='gray')
    # plt.axis('off')
    #
    #







