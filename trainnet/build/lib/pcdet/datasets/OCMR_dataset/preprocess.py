import os.path
import random
from pcdet.utils.read_ocmr import *
from pcdet.utils.utils import *
from tools.visual_utils.visualizer import *
import pickle
import h5py

# def read_kt_data(path):
#     '''
#     kData: [kx, ky, kz, coil, time, set, slice, rep, avg], coiled kt
#     set (velocity encoding), slice, repetition, and number of averages,
#     return: kt [time, coil, kx, ky], save as pickle
#     '''
#     kt_data, _ = read_ocmr(path)
#     # print('raw kdata is ',kt_data.shape)
#     slice = kt_data.shape[6] // 2
#
#     kt_data = kt_data[..., slice, :, :]  # take the middle slice
#     kt_data = np.mean(kt_data, axis=(-2,-1))  # average the k-space if average > 1
#     kt_data = np.squeeze(kt_data)  #[kx, ky, coil, time]
#     kt_data = kt_data.transpose(3, 2, 0, 1)  # [time, coil, kx, ky]
#     xt = kt2xt(kt_data)
#     xt = remove_oversampling(xt)
#     kt = img2kspace(xt)
#     print('new kt data is ', kt.shape)
#     return kt, slice

def read_kt_data(path):
    kt_file = h5py.File(kt_path, 'r')
    kt = np.array(kt_file['kspace']).squeeze(0).transpose(1, 0, 2, 3)
    kt_file.close()
    return kt

def read_coil_sens(path):
    '''
    coil_sens: # [slice, coil, frame(1), nx ,ny], H5 FILE
    Purpose: remove slice, choose middle slice
    return: [ncoil, nx, ny]
    '''
    coil_sen_file = h5py.File(path, 'r')
    coil_sens = np.array(coil_sen_file['coil_sens']).squeeze()  # [(slice), coil, kx, ky]
    if coil_sens.shape.__len__() == 4:
        slice_index = coil_sens.shape[0] // 2
        coil_sens = coil_sens[slice_index]  # [coil, kx, ky]
    print('coil_sens shape: ', coil_sens.shape)
    coil_sen_file.close()
    return coil_sens

def read_ref_img(path):
    '''
    ref_img: # [1,nt,nx,ny], H5 FILE
    return: xt [nt, nx, ny]
    '''
    data = h5py.File(path, 'r')
    ref_xt = data['reconstruction_sense']  # [1,nt,nx,ny]
    ref_xt = np.squeeze(ref_xt)
    print('ref_xt shape: ', ref_xt.shape)
    return ref_xt


if __name__ == '__main__':
    # kt_dir = '/home/vault/iwbi/shared/OCMR_data/raw'
    # coil_sens_dir = '/home/vault/iwbi/shared/OCMR_data/ESPIRiT/fs'
    # ref_img_dir = '/home/vault/iwbi/shared/OCMR_data/processed'
    # save_dir = '/home/hpc/iwbi/iwbi006h/TrainNet/data'

    kt_dir = '/media/liu/data/data/kt'
    coil_sens_dir = '/media/liu/data/data/coil_sens'
    save_dir = '/media/liu/data/data/preprocess'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    kt_names = sorted(os.listdir(kt_dir))
    all_pkl_data_path = []
    for kt_name in kt_names:
        if kt_name[:2] != 'fs':
            continue

        root_name = os.path.splitext(kt_name)[0][:10]
        kt_path = os.path.join(kt_dir, kt_name)
        kt = read_kt_data(kt_path)

        coil_sens_path = os.path.join(coil_sens_dir, root_name, 'coil_sens_avg.h5')
        coil_sens = read_coil_sens(coil_sens_path)


        # filter the shape of ny not the same
        if kt.shape[1:] != coil_sens.shape:
            print('shape diffenent: ', root_name, 'kt: ', kt.shape, ' coil_sens: ', coil_sens.shape)
            continue

        data = {'kt': kt,
                'coil_sens': coil_sens,
                }

        pkl_name = root_name + '.pkl'
        pkl_path = os.path.join(save_dir, pkl_name)

        all_pkl_data_path.append(pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


    split_percent = 0.7
    split_index = int(split_percent * len(all_pkl_data_path))
    train_infos = all_pkl_data_path[:split_index]
    test_infos = all_pkl_data_path[split_index:]

    train_infos_pkl = 'dataset_train_infos.pkl'
    test_infos_pkl = 'dataset_test_infos.pkl'

    with open(os.path.join(save_dir, train_infos_pkl), 'wb') as f:
        pickle.dump(train_infos, f)
    with open(os.path.join(save_dir, test_infos_pkl), 'wb') as f:
        pickle.dump(test_infos, f)

    print('proprocess finish')