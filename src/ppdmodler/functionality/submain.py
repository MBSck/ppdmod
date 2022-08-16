import sys
import corner
import numpy as np
import ultranest as un
import matplotlib.pyplot as plt
import scipy.stats

from ultranest.plot import cornerplot
from src.functionality.readout import ReadoutFits
from src.functionality.utilities import set_uvcoords, plot_txt
from src.functionality.genetic_algorithm import genetic_algorithm, decode
from src.models import Gauss2D, CompoundModel, UniformDisk
from src.functionality.fourier import FFT


# Shows the full np.arrays, takes ages to print the arrays
# np.set_printoptions(threshold=sys.maxsize)

def get_coords(im, W, H):
    h,w = im.shape
    x = np.arange(0,w+1,1) * W/w
    y = np.arange(0,h+1,1) * H/h
    return x,y

def im_interp(im, H,W):
    X = np.zeros(shape=(W,H))
    x, y = get_coords(im, W, H)
    for i,v in enumerate(X):
        y0_idx = np.argmax(y >i) - 1
        for j,_ in enumerate(v):
            # subtracting 1 because this is the first val
            # that is greater than j, want the idx before that
            x0_idx = np.argmax(x > j) - 1
            x1_idx = np.argmax(j < x)

            x0 = x[x0_idx]
            x1 = x[x1_idx]

            y0 = im[y0_idx, x0_idx - 1]
            y1 = im[y0_idx, x1_idx - 1]

            X[i,j] = scipy.interpn(y0, x0, y1, x1, j)
    return X

def im_resize(im,H,W):
    """Interpolates twice, once for x-direction then for y"""
    X_lin = im_interp(im, H,W)
    X = im_interp(X_lin.T, H,W)
    return X_lin, X.T

# NOTE: Fitting check
def straigth(x):
    xcoords = np.linspace(0, 100)
    return xcoords, xcoords*x[0] + x[1]

def chi_sq(x):
    return np.sum((data - straigth(x)[1])**2/(data*0.2)**2)

def test_chi_sq():
    bounds = [[0., 1.], [0, 5]]
    n_bits = 32
    n_pop, n_iter = 100, 1000
    r_cross = 0.85
    best, scores = genetic_algorithm(chi_sq, bounds, n_bits, n_iter, n_pop,
                                     r_cross, 1./(n_bits*len(bounds)))
    decoded = decode(bounds, n_bits, best)
    return decoded

def main():
    ...

if __name__ == "__main__":
    main()
