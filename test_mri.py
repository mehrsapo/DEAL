from batch_wrapper import test
import argparse

def test_mri(device='cuda:0'):
    modality = 'mri'
    method = 'deal'
    n_hyperparameters = 2

    tol = 1e-8
    max_iter = 1000

    if __name__ == "__main__":
        # argpars
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', type=str, default="cuda:3")

        device = parser.parse_args().device

        EXP_NAMES = ['DEAL']
        DATA_TYPES = ['pd', 'pdfs']
        opt = '2'
        noise_sd = 0.002

        if (opt == '1'):
            coil_type = 'single'
            acc = 2
            cf = 0.16

        elif (opt == '2'):
            coil_type = 'single'
            acc = 4
            cf = 0.08

        elif (opt == '3'):
            coil_type = 'multi'
            acc = 4
            cf = 0.08

        elif (opt == '4'):
            coil_type = 'multi'
            acc = 8
            cf = 0.04

        
        for data_type in DATA_TYPES:
            for exp_name in EXP_NAMES:
                job_name = f"{modality}_coiltype_{coil_type}_acc_{acc}_cf_{cf}_noisesd_{noise_sd}_datatype_{data_type}_deal_DEAL_v2"
                test(method = method, modality = modality, job_name=job_name, coil_type=coil_type, acc=acc, cf=cf, noise_sd=noise_sd, data_type=data_type, device=device, model_name = exp_name, n_hyperparameters=n_hyperparameters, tol=tol, max_iter=max_iter, crop=True)



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--d', type=str, default='cuda:0')
    device = argparser.parse_args().d
    test_mri(device)