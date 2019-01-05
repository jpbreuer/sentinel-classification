import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import numpy as np

# DEFINE SUPER MODEL
class Zoolander:
    def __init__(self, num_classes, input_shape=(32, 32, 18)):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        self.model = model

    def predict(self, inputs):
        if type(inputs)==type([]):
            inputs = np.array(inputs)
        predictions = self.model.predict(inputs)

        return predictions

    def load_weights(self, weights):
        self.model.load_weights(weights)

###--- DEFINE URBAN MODEL --->

class UrbanModel:
    def __init__(self, num_classes, input_shape=(32, 32, 10)):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        self.model = model

    def predict(self, inputs):
        if type(inputs)==type([]):
            inputs = np.array(inputs)
        predictions = self.model.predict(inputs)

        return predictions

    def load_weights(self, weights):
        self.model.load_weights(weights)  

###--- DEFINE VEG MODEL --->

class VegModel:
    def __init__(self, num_classes, input_shape=(32, 32, 11)):
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        self.model = model

    def predict(self, inputs):
        if type(inputs)==type([]):
            inputs = np.array(inputs)
        predictions = self.model.predict(inputs)

        return predictions

    def load_weights(self, weights):
        self.model.load_weights(weights)  

