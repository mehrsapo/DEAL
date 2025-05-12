import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from natsort import os_sorted

from forward_operators.forward_operators_SR import * 
import json

from deal import * 

from utils_dpir.utils_image import * 

import time 

def SR(hparams):
   
    device = hparams.device
    times = list()
    path_ckp = "trained_models/deal_color.pth"
    ckp = torch.load(path_ckp, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

    model = DEAL(color=True)
    model.to(device)
    model.load_state_dict(ckp['state_dict'])
    model.eval()

    hparams.sigma_denoiser = 15
    hparams.lamb = 0.1 + 0.035 * hparams.noise_level_img ** 2  # 0.1 0.28 2.5 5.5 for 0 2.55 7.65 12.75 # table 2
    # SR specific hyperparameters
    eps_in = 5e-7
    eps_out = 1e-5
    classic = True   # for gaussian kernels, False for bicubic kernels
    save_LR = False # create folders if not exist
    save_GT = False
    fourier = True
    hparams.degradation_mode = 'SR'

    if hparams.noise_level_img == 0:
        noise_id = 0
    if hparams.noise_level_img == 2.55:
        noise_id = 255
    if hparams.noise_level_img == 7.65:
        noise_id = 765
    if hparams.noise_level_img == 12.75:
        noise_id = 1275
    

    # Set input image paths
    input_path = os.path.join(hparams.dataset_path, hparams.dataset_name)
    input_path = os.path.join(input_path, os.listdir(input_path)[0])
    input_paths = os_sorted([os.path.join(input_path, p) for p in os.listdir(input_path)])

    # Load the 8 blur kernels
    if classic:
        kernel_path = "kernels/kernels_12.mat"
        k_list = range(4)
    
    else:
        kernel_path = "kernels/kernels_bicubicx234.mat"
        k_list = range(1)

    print(kernel_path)
    kernels  = hdf5storage.loadmat(kernel_path)['kernels']
    
    print(kernels.shape)
    counter = 0
    # Kernels follow the order given in the paper (Table 3)
    psnr_list = []
    psnr_list_deal = []

    print(
        '\n DEAL super-resolution with image sigma:{:.3f}, model sigma:{:.3f}, lamb:{:.3f} \n'.format(
            hparams.noise_level_img, hparams.sigma_denoiser, hparams.lamb))


    for k_index in k_list: 

        if not classic:  # for bicubic degradation
            k_index = hparams.sf - 2

        psnr_k_list_deal = []
        psnrY_k_list = []

        k = kernels[0, k_index].astype(np.float64)

        for i in range(min(len(input_paths),hparams.n_images)): 

            print('__ kernel__',k_index, '__ image__',i)

            ## load image
            input_im_uint = imread_uint(input_paths[i])
            if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
               input_im_uint = crop_center(input_im_uint, hparams.patch_size,hparams.patch_size)
            # Degrade image
            input_im_uint = modcrop(input_im_uint, hparams.sf)
            input_im = np.float32(input_im_uint / 255.)
            if classic: 
                blur_im = numpy_degradation(input_im, k, hparams.sf)
            else: 
                blur_im = imresize_np(input_im, 1/hparams.sf)

            np.random.seed(seed=0)
            noise = np.random.normal(0, hparams.noise_level_img/255., blur_im.shape)
            blur_im += noise

            kernel_torch =  torch.tensor(k)[None, None].float().to(device)

            if not classic and not fourier: 
                kernel_torch = kernel_torch[..., :-(hparams.sf-1)*2, :-(hparams.sf-1)*2]

            kk = kernel_torch.repeat(1, 3, 1, 1).float()
            mirrored_filter = torch.rot90(kk, 2)
            pad_mirrored_flter = torch.nn.functional.pad(mirrored_filter, (0, input_im.shape[1]-kk.size(3), 0, input_im.shape[0]-kk.size(2)))
            rolled_pm_filter = torch.roll(torch.roll(pad_mirrored_flter, -(kk.size(2)-1)//2, 2), -(kk.size(3)-1)//2, 3)
            fft_filters = torch.fft.fft2(rolled_pm_filter)

            if not fourier:
                H = lambda x: torch_degradation(x, kernel_torch, sf=hparams.sf)
                Ht = lambda z: torch_degradation_transpose(z, kernel_torch, sf=hparams.sf)
            else:
                H = lambda x: torch_degradation_fourier(x, fft_filters, sf=hparams.sf)
                Ht = lambda z: torch_degradation_transpose_fourier(z, fft_filters, sf=hparams.sf)

            y_torch = torch.tensor(blur_im).transpose(0, 2).transpose(1, 2)[None, ...].float().to(device)
            counter = counter + 1
            if save_LR: 
                if classic: 
                    imsave(single2uint(blur_im).squeeze(), 'SR_LR/' + 'X' + str(hparams.sf)+'/'+ str(noise_id) + '/' + str(counter) + 'x'+str(hparams.sf)+'.png')
                if classic: 
                    imsave(single2uint(blur_im).squeeze(), 'SR_LR/' + 'bicubic/X' + str(hparams.sf)+'/'+ str(noise_id) + '/' + str(counter) + 'x'+str(hparams.sf)+'.png')
            
            if save_GT: 
                if classic: 
                    imsave(single2uint(input_im).squeeze(), 'SR_GT/' + 'classic/' + str(counter) + '.png')
                else:
                    imsave(single2uint(input_im).squeeze(), 'SR_GT/' + 'bicubic/' + str(counter) + '.png')

            t1 = time.time()
            out_deal = model.solve_inverse_problem(y_torch, H, Ht, hparams.sigma_denoiser, hparams.lamb, verbose=False, eps_in=eps_in, eps_out=eps_out)
            t2 = time.time()
            times.append(t2-t1)
            psnr_deal = psnr(input_im, tensor2array(out_deal.cpu()))
            print('PSNR DEAL: {:.2f}dB'.format(psnr_deal))

            psnr_k_list_deal.append(psnr_deal)
            psnr_list_deal.append(psnr_deal)

        avg_k_psnr_deal = np.mean(np.array(psnr_k_list_deal))
        print('avg RGB psnr on kernel deal {}: {:.2f}dB'.format(k_index, avg_k_psnr_deal))

    print(np.mean(np.array(psnr_list)))
    print('DEAL', np.mean(np.array(psnr_list_deal)), hparams.sigma_denoiser, hparams.lamb, hparams.noise_level_img)
    print('DEAL Time Avg', np.mean(np.array(times)), hparams.sigma_denoiser, hparams.lamb, hparams.noise_level_img)
    return np.mean(np.array(psnr_list))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--n_images', type=int, default=68)
    parser.add_argument('--noise_level_img', type=float, default=0)
    parser.add_argument('--dataset_path', type=str, default='sr_datasets')
    parser.add_argument('--dataset_name', type=str, default='CBSD68')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--sf', type=int, default=2)
    parser.add_argument('--kernel_path', type=str, default=os.path.join('kernels', 'kernels_12.mat'))
    hparams = parser.parse_args()
    SR(hparams)

