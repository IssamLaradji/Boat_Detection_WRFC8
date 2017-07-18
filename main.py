
import numpy as np

from skimage.color import rgb2gray
from skimage import data, io, filters
import os
from skimage import morphology as morph

from skimage.morphology import convex_hull_image
from skimage.filters import gaussian
from sklearn.utils import shuffle
from skimage.transform import resize
from skimage import img_as_float
from glob import glob
import argparse, sys
import pylab as plt
parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)
from mlkit import bc_utils as bc 
from mlkit.pytorch_kit import models as tm
from mlkit import image_utils as iu

data_home = "../../Datasets/" 

def show(m, i, xx=None, image=False):
    if xx is None:
        xh = m.get_heatmap(X[i:i+1])
        print "Max activation: %f" % xh.max()
        if image:
            iu.show(xh, X[i], y[i])
    else:
        xh = m.get_heatmap(xx[i:i+1])
        print "Max activation: %f" % xh.max()
        if image:
            iu.show(xh, xx[i], y[i])


if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--lake_name', default="Yellow_docks_1",
                        choices=["Yellow_docks_1", "Como_Lake_2"])  
    parser.add_argument('-w','--weakly_labeled', type=int, default=1)
    parser.add_argument('-r','--reset', type=int, default=0)  
    io_args = parser.parse_args()
    reset = io_args.reset

    # GET DATA
    lake = io_args.lake_name
    laked = bc.load_config(lake, data_home)
    weakly_labeled = io_args.weakly_labeled

    #------- 1. TRAIN SET
    train_set = np.arange(laked["n_train"])
    fname = "_train_%s_%d.npy" % (lake, weakly_labeled)
    if os.path.exists("X" + fname) and not reset:
        X = np.load("X" + fname)
        y = np.load("y" + fname)
        print "%s: X, y exists..." % fname 
    else:
        X, y = bc.get_imgList(train_set, bg=True, laked=laked, 
                                weakly_labeled=weakly_labeled)
        np.save("X" + fname, X)
        np.save("y" + fname, y)

        print "%s: X, y saved..." % fname

    train_fi = np.reshape(y, (y.shape[0],-1)).sum(axis=1) > 1
    train_nf = np.logical_not(train_fi)

    Xfi, yfi = X[train_fi], y[train_fi]
    Xnf, ynf = X[train_nf], y[train_nf]

    #------- 2. VALID SET
    fname = "_valid_%s_%d.npy" % (lake, weakly_labeled)
    valid_set = np.arange(laked["n_train"], laked["n_train"]+laked["n_valid"])
    if os.path.exists("X" + fname) and not reset:
        Xv = np.load("X" + fname)
        yv = np.load("y" + fname)
        print "%s: Xv, yv exists..." % fname
    else:

        Xv, yv = bc.get_imgList(valid_set, bg=True, laked=laked, 
                                weakly_labeled=weakly_labeled)
        np.save("X" + fname, Xv)
        np.save("y" + fname, yv)
        print "%s: Xv, yv saved saved..." % fname

    n_rows, n_cols = X.shape[2], X.shape[3]

    # ---------- 4. TRAIN MODEL
    #model = mt.SmallUNet(n_channels=3, n_classes=1, loss_name="binary_crossentropy")
    model = tm.LocalizePoint(n_channels=3, n_outputs=1, loss_name="bce_localize")
    # xx = X[:50,:, 130:210, 400:480]
    # yy = y[:50,:, 130:210, 400:480]
    # x = xx[20:40]
    # y = np.zeros(20); y[0] = 1; y[13] = 1; y[8]=1
    # #model = mt.Localize(n_channels=3, n_outputs=2)
    # # model = mt.Localize_2(n_channels=3, n_outputs=1)
    # # model.forward_pass(x)
    # # model.get_heatmap(x)

    
    y[y==-1] = 0
    X = X[:,:,:300,:330]
    y = y[:,:,:300,:330]
    Xv = Xv[:,:,:300,:330]
    import pdb; pdb.set_trace()  # breakpoint bac9388b //

    model.fit(X[:50], y[:50], epochs=100, batch_size=50)