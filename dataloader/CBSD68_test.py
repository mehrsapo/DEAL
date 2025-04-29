import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random

from datasets import load_dataset

random.seed(1)

class CBSD68_test(Dataset):

    def __init__(self):
        super(Dataset, self).__init__()
        self.dataset = load_dataset("deepinv/CBSD68", split='train')

    def __len__(self):
        return 68

    def __getitem__(self, idx):
        data =np.array(self.dataset.__getitem__(idx)['png']) / 255.
        data = torch.tensor(data).float() # s.transpose(0, 2).transpose(1, 2)
        return data