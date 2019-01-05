import sys
import pandas
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

def viz_generator(gen):
    for batch in gen:
        viz_batch(batch)

def viz_batch(batch, classes=None):
    images, labels = batch
    for img, label in zip(images, labels):
        if classes:
            print(classes[np.argmax(label)])
        fig = plt.figure()
        plt.imshow(4*img)
        plt.draw()
        plt.pause(1) # <-------
        inpt = input("<Hit q To Close>")
        print(inpt)
        if inpt=='q':
            plt.close(fig)
            sys.exit(0)
        plt.close(fig)
