import h5py
import numpy as np

def load_dataset(path):
    fid = h5py.File(path,'r')
    data_s1 = fid['sen1']
    data_s2 = fid['sen2']
    labels = fid['label']

    return data_s1, data_s2, labels


