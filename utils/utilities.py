import torch
import math
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr 
from skimage.metrics import structural_similarity as compare_ssim


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return PSNR/Img.shape[0]


def batch_SSIM(img, imclean, data_range):
    img = torch.transpose(img, 1, 3)
    imclean = torch.transpose(imclean, 1, 3)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += compare_ssim(Iclean[i,:,:,0], Img[i,:,:,0], data_range=data_range, multichannel=True)
    return SSIM/Img.shape[0]

def compute_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = compare_psnr(Iclean[0,:,:,:], Img[0,:,:,:], data_range=data_range)
    return PSNR
    
def crop_center(img,cropx,cropy):
    y,x = img.shape[0],img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx, :]

def center_crop(data, shape):
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]