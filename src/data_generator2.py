import h5py from keras.utils.io_utils import HDF5Matrix
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

class AugumentedHDF5Matrix(HDF5Matrix):
    """Wraps HDF5Matrixs with image augumentation."""

    def __init__(self, image_datagen, seed, *args, **kwargs):
        self.image_datagen = image_datagen
        self.seed = seed
        self.i = 0
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        x = super().__getitem__(key)
        self.i += 1
        x = x[:,:,:]
        bgr = x[:,:,0:3]
        rgb = bgr[...,::-1]
        if len(x.shape) == 3:
            return self.image_datagen.random_transform(
                rgb, seed=self.seed + self.i)
        else:
            return np.array([
                self.image_datagen.random_transform(
                    xx, seed=self.seed + self.i) for xx in x
            ])

class DataSet:
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

    data_gen_args = dict( 
            rotation_range=90.
            # width_shift_range=0.05,
            # height_shift_range=0.05,
            # zoom_range=0.2,
            # channel_shift_range=0.005,
            # horizontal_flip=True,
            # vertical_flip=True,
            # fill_mode='constant',
            # data_format="channels_last",
    )

    def __init__(self, path, classes=None):
        self.data_s1, self.data_s2, self.labels = self.load_data_fromh5(path)

    def load_data_fromh5(self, path):
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        data_sen1 = AugumentedHDF5Matrix(image_datagen, 0, path,'sen1')
        data_sen2 = AugumentedHDF5Matrix(image_datagen, 0, path,'sen2')
        labels = HDF5Matrix(path,'labels')
        return data_sen1, data_sen2, labels

    def generator(self,
                shuffle=True, 
                seed = 10,
                batch_size = 64,
                augment = True):
                # target_size=(112,112),
                # color_mode='RGB',
                # preprocessing = True,
                # augmentation = False):

        if augment:
            image_datagen = ImageDataGenerator(**self.data_gen_args)
        else: 
            image_datagen = ImageDataGenerator()

        image_generator = image_datagen.flow(
            self.data_s2, y=self.labels, 
            shuffle=shuffle,
            seed=seed, 
            batch_size=batch_size,
        )

        return image_generator
