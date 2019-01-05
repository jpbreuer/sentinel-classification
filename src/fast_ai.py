# This file contains all the main external libs we'll use
from fastai.imports import *
# from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


import h5py
import torch
import torch.utils.data as data

class H5Dataset(data.Dataset):

    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('sen2')
        self.target = h5_file.get('labels')

    def __getitem__(self, index):            
        x = self.data[index, :, :, 0:3]
        x = x[...,::-1] #from BGR to RGB
        y = self.target[index]
        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float())

    def __len__(self):
        return self.data.shape[0]



import IPython
IPython.embed()