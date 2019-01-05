import os
import sys

#Ext libraries
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#Custom libraries
from data_generator import DataSet
from model import Model

# PATHS
DATA_PATH = '../data/'
WEIGHTS_NAME = 'tst.hdf5'
TRAIN_PATH = os.path.join(DATA_PATH, 'subset_training.hdf5')
VALID_PATH = os.path.join(DATA_PATH, 'subset_validation.hdf5')

#HYPER-PARAMETERS
NUM_EPOCHS = 260
BATCH_SIZE = 32
LEARNING_RATE = 0.01
MULTI_GPU = False
LOAD_WEIGHTS = True
NUM_GPUS = 8


# DATAGENERATORS
train_dataset = DataSet(TRAIN_PATH)

val_dataset = DataSet(VALID_PATH)

num_classes = len(val_dataset.classes)
classifier = Model(num_classes)
print("Model assambled")
import IPython
IPython.embed()

train_gen = train_dataset.generator()
print("Loaded train")
val_gen = val_dataset.generator()
print("Loaded validation")

# OPTIMIZER
sgd = SGD(lr=LEARNING_RATE, decay=0, momentum=0.9, nesterov=True)

# CONVERTING TO GPU
if MULTI_GPU:
    model = keras.utils.multi_gpu_model(classifier.model, gpus=NUM_GPUS,
            cpu_merge=True, cpu_relocation=False)

# COMPILING MODEL
classifier.model.compile(loss='categorical_crossentropy',
        metrics=['categorical_accuracy', 'categorical_accuracy'],
        optimizer=sgd)

#Load model weights
if LOAD_WEIGHTS:
    classifier.model.load_weights(WEIGHTS_NAME)

# DEFINING checkpoint callbacks
checkpoint = ModelCheckpoint(WEIGHTS_NAME,
        monitor='categorical_accuracy',
        verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# TRAINING MODEL
len_train_gen = int(np.floor(len(train_dataset.data_s1) / BATCH_SIZE))
len_val_gen = int(np.floor(len(val_dataset.data_s2) / BATCH_SIZE))
classifier.model.fit_generator(
        train_gen,
        steps_per_epoch=len_train_gen/BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=callbacks_list,
        validation_data=val_gen,
        validation_steps=len_val_gen/BATCH_SIZE)

import IPython
IPython.embed()
