import sys
import argparse
import torch
import os
import json
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
from utils.utilities import center_crop
from tqdm import tqdm
from deal import DEAL

from forward_operators_MRI import get_operators_mri

def DEAL_Recon(y, model, lmbda=1, sigma=1, H=None, Ht=None, op_norm=1, x_gt=None, **kwargs):
    """ """
    max_iter = kwargs.get('max_iter', 1000)
    eps_in = kwargs.get('tol', 1e-8)
    eps_out = 1e-5
    crop = kwargs.get('crop', False)
    
    x = torch.clone(Ht(y)).zero_().detach()
    x_old = torch.clone(x)
    
    pbar = tqdm(range(max_iter))

    model.cal_scaling(torch.tensor([[sigma]]).float().to(y.device))
    model.lmbda = torch.tensor([[lmbda]]).to(y.device).float()
    
    for i in pbar:
        with torch.no_grad():
            model.cal_mask(x)
            x, _ = model.cg(Ht(y), x_old, 1000, eps=eps_in, H=H, Ht=Ht)
            res = torch.norm(x-x_old)/(torch.norm(x_old))
            x_old = x.clone()
            
            if (res < eps_out).all():
                break

        
        if x_gt is not None:
            if crop:
                x_crop = center_crop(x, [320,320])
                psnr_ = psnr(x_crop, x_gt, data_range=1)
                ssim_ = ssim(x_crop, x_gt, data_range=1)
            else:
                psnr_ = psnr(x, x_gt, data_range=1.0)
                ssim_ = ssim(x, x_gt, data_range=1.0)
            pbar.set_description(f"psnr: {psnr_:.2f} | ssim: {ssim_:.4f} | res: {res:.2e}")
        else:
            psnr_ = None
            ssim_ = None
            pbar.set_description(f"res: {res:.2e}")


    return(x, psnr_, ssim_, i)



def get_reconstruction_map(method, modality, device="cuda:0", **opts):
    model_name = opts.get('model_name', 'DEAL')

    if model_name is None:
        raise ValueError("Please provide a model_name for the wcrr model. It is the name of the folder corrsponding to the trained model.")
    
    path_ckp = "trained_models/deal_gray.pth"
    ckp = torch.load(path_ckp, map_location={'cuda:0':device,'cuda:1':device,'cuda:2':device,'cuda:3':device})

    model = DEAL(color=False)
    model.to(device)
    model.load_state_dict(ckp['state_dict'])
    model.eval()
    model.W1.spectral_norm()

    if modality == 'mri':
        def reconstruction_map(y, mask, smap, p1, p2, x_gt=None):
            with torch.no_grad():
                H, Ht, _ = get_operators_mri(mask, smap, device=device)
                x, psnr_, ssim_, n_iter = DEAL_Recon(y.cfloat(), model.float(), lmbda=p1, sigma=p2, H=H, Ht=Ht, x_gt=x_gt.float(), **opts)
            return(x, psnr_, ssim_, n_iter)
    
    return(reconstruction_map)


