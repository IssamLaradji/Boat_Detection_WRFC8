import numpy as np

from skimage.color import rgb2gray
from skimage import data, io, filters
import os
from skimage import morphology as morph

from skimage import img_as_float
from skimage.filters import gaussian
from sklearn.utils import shuffle
from skimage.transform import resize
from skimage import img_as_float
from glob import glob
import argparse, sys
import pylab as plt
from skimage.io import imread
from model import BoatDetector
import torch
import pylab as plt

def last2first(X):
    # CHANNELS LAST
    if X.ndim == 3:
        return np.transpose(X, (2,0,1))
    if X.ndim == 4:
        return np.transpose(X, (0,3,1,2))

def first2last(X):
    # CHANNELS FIRST
    if X.ndim == 3:
        return np.transpose(X, (1,2,0))
    if X.ndim == 4:
        return np.transpose(X, (0,2,3,1))

if __name__ == "__main__":
    # 1. LOAD IMAGES
    X = []
    for img_id in range(4):
        X += [img_as_float(imread("images/%d.png" % img_id))]

    X = last2first(np.array(X))
    n_rows, n_cols = X.shape[2:]

    # 2. LOAD MODEL
    model = BoatDetector(n_channels=3, n_outputs=1)
    model.load_state_dict(torch.load("model_weights.pth", map_location=lambda storage, loc: storage))

    # 3. PREDICT
    y_pred = model.predict(X)
    y_pred = np.squeeze(y_pred)

    # 4. SAVE PREDICTIONS
    for i in range(len(y_pred)):
        plt.subplot(1, 2, 1)
        plt.imshow(first2last(X[i]))

        plt.subplot(1, 2, 2)
        plt.imshow(first2last(X[i]))
        plt.imshow(resize(y_pred[i], (n_rows, n_cols)), alpha=0.5)

        plt.tight_layout()
        plt.savefig("results/%d.png" % i)

        plt.show()
        plt.close()
    import pdb; pdb.set_trace()  # breakpoint bd2e165c //
