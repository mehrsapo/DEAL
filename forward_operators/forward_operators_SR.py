import torch 
from scipy import ndimage
import numpy as np  
import cv2
# only for 256x256 images, easy to fix for other sizes, need to change dd in upsample function

import torch 

import torch 

def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    out = x[..., st::sf, st::sf]
    return out

def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    if sf == 3:
        dd = 255
    else:
        dd = 256
    z = torch.zeros((1, 3, dd, dd)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z

def periodic_pad_transpose(tensor: torch.Tensor, pad_size) -> torch.Tensor:
        """
        Adjoint of the periodic padding operation which amounts to a special type of cropping.

        :param tensor: input of shape [..., H, W] to be cropped
        :return: cropped tensor of shape [..., H - pad_h, W - pad_w]
        """
        sz = [tensor.size(0), tensor.size(1), tensor.size(2), tensor.size(3)]
        out = tensor.clone()
        # Top
        if pad_size[1] != 0:
            out[..., pad_size[0]:pad_size[0] + pad_size[1], :] += out[..., sz[-2]-pad_size[1]::, :]
        # Bottom
        if pad_size[0] != 0:
            out[..., -pad_size[0] - pad_size[1]:sz[-2]-pad_size[1], :] += out[..., 0:pad_size[0], :]
        # Left
        if pad_size[3] != 0:
            out[..., pad_size[2]:pad_size[2] + pad_size[3]] += out[..., sz[-1]-pad_size[3]::]
        # Right
        if pad_size[2] != 0:
            out[..., -pad_size[2] - pad_size[3]:sz[-1]-pad_size[3]] += out[..., 0:pad_size[2]]
        if pad_size[1] == 0:
            end_h = sz[-2] + 1
        else:
            end_h = sz[-2] - pad_size[1]
        if pad_size[3] == 0:
            end_w = sz[-1] + 1
        else:
            end_w = sz[-1] - pad_size[3]
        out = out[..., pad_size[0]:end_h, pad_size[2]:end_w]
        return out


def imfilter(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    '''
    x = torch.nn.functional.pad(x, ((k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2, (k.shape[-2] - 1) // 2, (k.shape[-2] - 1) // 2), mode='circular')
    x = torch.nn.functional.conv2d(x, k, groups=x.shape[1])
    return x


def imfilter_transpose(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    '''
    x = torch.nn.functional.conv_transpose2d(x, k, groups=x.shape[1])
    pd = (k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2, (k.shape[-2] - 1) // 2, (k.shape[-2] - 1) // 2
    x = periodic_pad_transpose(x, pad_size=(pd))
    return x


def torch_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: NxCHxW image, [0, 1]/[0, 255]
        k: kernel, 1x1xhxw
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = imfilter(x, k.repeat(3, 1, 1, 1).float())
    x = downsample(x, sf=sf)
    return x



def torch_degradation_transpose(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: NxCHxW image, [0, 1]/[0, 255]
        k: kernel, 1x1xhxw
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = upsample(x, sf)
    x = imfilter_transpose(x, k.repeat(3, 1, 1, 1).float())
    return x

def imfilter_fourier(x, k_fft):
    '''
    x: image, NxcxHxW
    k_fft: fft of kernel, cx1xhxw
    '''
    return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * k_fft))

def imfilter_transpose_fourier(x, k_fft):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    '''
    return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * torch.conj(k_fft)))

def torch_degradation_fourier(x, k_fft, sf=3):
    ''' blur + downsampling

    Args:
        x: NxCHxW image, [0, 1]/[0, 255]
        k: kernel, 1x1xhxw
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = imfilter_fourier(x, k_fft)
    x = downsample(x, sf=sf)
    return x



def torch_degradation_transpose_fourier(x, k_fft, sf=3):
    ''' blur + downsampling

    Args:
        x: NxCHxW image, [0, 1]/[0, 255]
        k: kernel, 1x1xhxw
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = upsample(x, sf)
    x = imfilter_transpose_fourier(x, k_fft)
    return x

def numpy_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double, positive
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    # x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


def psnr(img1,img2) :
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2)**2)
    return 20 * np.log10(1. / np.sqrt(mse))


def array2tensor(img):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

def single2uint(img):
    return np.uint8((img.clip(0, 1)*255.).round())

def imsave(img_path,img):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)

def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:int(H-H_r), :int(W-W_r), :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img

def crop_center(img,cropx,cropy):
    y,x = img.shape[0],img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]




def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


def tensor2array(img):
    img = img.cpu()
    img = img.squeeze().detach().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img