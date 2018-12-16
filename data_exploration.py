#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 21:52:47 2018

@author: jpbreuer
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

base_dir = "/home/jpbreuer/alibaba-ai/"
path_training = os.path.join(base_dir, "training.h5")
path_test = os.path.join(base_dir, "validation.h5")

#training_data
train = h5py.File(path_training,'r')
#validation_data = h5py.File(path_test,'r')

label = train['label']
sen1 = train['sen1']
sen2 = train['sen2']
#s1_training = training_data['sen1']
#s2_training = training_data['sen2']
#label_training = training_data['label']
#
#s1_validation = validation_data['sen1']
#s2_validation = validation_data['sen2']
#label_validation = validation_data['label']
#
#print("shape for each channel.")
#print(s1_training.shape)
#print(s2_training.shape)
#print(label_training.shape)
#print("-" * 60)
#print("validation part")
#print(s1_validation.shape)
#print(s2_validation.shape)
#print(label_validation.shape)
#
#print("show class distribution")
label_qty = np.sum(label, axis=0)
position,pclass = np.nonzero(label)
#print(label_qty)
#plt.plot(label_qty)
#plt.title("class distribution")

c0index = np.where(pclass==0)[0]
s1c0 = list()
for item in list(range(1000)):
    s1c0.append(sen1[c0index[item]])

#%% 

# visualization, plot the first pair of Sentinel-1 and Sentinel-2 patches of training.h5
import matplotlib.pyplot as plt

plt.subplot(121)
plt.imshow(10*np.log10(s1_training[0,:,:,4]),cmap=plt.cm.get_cmap('gray'));
plt.colorbar()
plt.title('Sentinel-1')

plt.subplot(122)
plt.imshow(s2_training[0,:,:,1],cmap=plt.cm.get_cmap('gray'));
plt.colorbar()
plt.title('Sentinel-2')

plt.show()

### as you can see, it is difficult to identify the image as a class by human.

#%% 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

## for a small dataset train and test

train_s1 = np.array(fid_training['sen1'])
train_s2 = np.array(fid_training['sen2'])
train_label = np.array(fid_training['label'])
train_y = np.argmax(train_label, axis=1)


validation_s1 = np.array(fid_validation['sen1'])
validation_s2 = np.array(fid_validation['sen2'])
validation_label = np.array(fid_validation['label'])
validation_y = np.argmax(validation_label, axis=1)


n = train_s1.shape[0]
train_s1 = train_s1.reshape((n, -1))
train_s2 = train_s2.reshape((n, -1))
train_X = np.hstack([train_s1, train_s2])

n = validation_s1.shape[0]
validation_s1 = validation_s1.reshape((n, -1))
validation_s2 = validation_s2.reshape((n, -1))
validation_X = np.hstack([validation_s1, validation_s2])

clf = RandomForestClassifier()
clf.fit(train_X, train_y)
pre_val_y = clf.predict(validation_X)

print(classification_report(validation_y, pre_val_y))

#%% 
### simple classification example
### Training part using batch

from sklearn.linear_model import SGDClassifier

train_s1 = s1_training
train_s2 = s2_training
train_label = label_training
clf = SGDClassifier()

train_y = np.argmax(train_label, axis=1)
classes = list(set(train_y))
batch_size = 100
n_sampels = train_s1.shape[0]

for i in range(0, n_sampels, batch_size):
    ## this is an idea for batch training
    ## you can relpace this loop for deep learning methods
    if i % batch_size * 10 == 0:
        print("done %d/%d" % (i, n_sampels))
    start_pos = i
    end_pos = min(i + batch_size, n_sampels)
    train_s1_batch = np.asarray(train_s1[start_pos:end_pos, :, :, :])
    train_s2_batch = np.asarray(train_s2[start_pos:end_pos, :, :, :])
    cur_batch_size = train_s2_batch.shape[0]
    train_s1_batch = train_s1_batch.reshape((cur_batch_size, -1))
    train_s2_batch = train_s2_batch.reshape((cur_batch_size, -1))
    train_X_batch = np.hstack([train_s1_batch, train_s2_batch])
    label_batch = train_y[start_pos:end_pos]
    clf.partial_fit(train_X_batch, label_batch, classes=classes)
    
#%% 
### make a prediction on validation
pred_y = []
train_val_y = np.argmax(label_validation, axis=1)
batch_size = 100
n_val_samples = s2_validation.shape[0]
for i in range(0, n_val_samples, batch_size):
    start_pos = i
    end_pos = min(i + batch_size, n_val_samples)
    val_s1_batch = np.asarray(s1_validation[start_pos:end_pos, :, :, :])
    val_s2_batch = np.asarray(s2_validation[start_pos:end_pos, :, :, :])
    cur_batch_size = val_s2_batch.shape[0]
    val_s1_batch = val_s1_batch.reshape((cur_batch_size, -1))
    val_s2_batch = val_s2_batch.reshape((cur_batch_size, -1))
    val_X_batch = np.hstack([val_s1_batch, val_s2_batch])
    tmp_pred_y = clf.predict(val_X_batch)
    pred_y.append(tmp_pred_y)
pred_y = np.hstack(pred_y)

from sklearn.metrics import classification_report
print(classification_report(train_val_y, pred_y))
