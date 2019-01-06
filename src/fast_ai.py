# This file contains all the main external libs we'll use
# from fastai.imports import *
# # from fastai.transforms import *
# from fastai.conv_learner import *
# from fastai.model import *
# from fastai.dataset import *
# from fastai.sgdr import *
# from fastai.plots import *
from fastai.layers import simple_cnn
from fastai.basic_train import Learner

import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from fastai.vision import ImageDataBunch
from fastai.vision import *

class H5Dataset(Dataset):
    classes = ['compact-highrise',
           'compact-midrise',
            'compact-lowrise',
            'open-high-rise',
            'open-midrise',
            'open-lowrise',
            'lightweight-lowrise',
            'large-lowrise',
            'sparsely-built',
            'heavy-industry',
            'dense-trees',
            'scattered-trees',
            'bush-and-scrub',
            'low-plants',
            'bare-rock-or-paved',
            'bare-soild-or-sand',
            'water']
    c = len(classes)

    loss_func = torch.nn.CrossEntropyLoss()

    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('sen2')
        self.target = h5_file.get('labels')

    def __getitem__(self, index):            
        x = self.data[index, :, :, 0:3].astype(np.float32)
        x = x[...,::-1] #from BGR to RGB
        x = x.transpose(2,0,1)
        # y = self.target[index].astype(np.float32)
        y = np.argmax(self.target[index])
        return (torch.from_numpy(np.copy(x)).float(),
                torch.tensor(y))

    def __len__(self):
        return self.data.shape[0]

dataset_train = H5Dataset('../data/subset_training.hdf5')
dataset_val = H5Dataset('../data/subset_validation.hdf5')
data = ImageDataBunch.create(dataset_train, dataset_val)

learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(150,1e-2)

import IPython
IPython.embed()