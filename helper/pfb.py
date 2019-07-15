import numpy as np
import scipy
import matplotlib
import sys
import os
matplotlib.use('Agg')
from scipy.signal import firwin, freqz, lfilter
# This script is taken from https://github.com/telegraphic/pfb_introduction
# Danny C. Price, Spectrometers and Polyphase Filterbanks in Radio Astronomy, 2016. arXiv 1607.03579
# URL: http://arxiv.org/abs/1607.03579
# It repository is recommended to understand the process of the Polyphase Filterbank.
# However, for our context some function where not necessary and removed.
# Furthermore, the input is not generated randomly, but expected to be available at the files
# 1. "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/coeff"
# 2. "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/input"
# Where input is a list of floating numbers (newline for each number) of the size (nSpectra + nTaps - 1 ) x nChannels
# and coeff is a list of floating numbers (newline for each number) of the size nTaps x nChannels.
# The script expects 4 arguments passed when calling:
# 1. The timestamp of the input data (to compare it to the CUDA script)
# 2. The number of Taps
# 2. The number of Channels
# 2. The number of Spectra
# If you do not want to compare the results to the CUDA script please consider to use the original from github.

# Please consider that this script is customized to the path "/mnt/c/Users/b98/PFB_OUTPUT/".
# You quiet likely need to adjust that path to the place where you are storing the files.


def pfb_fir_frontend(x, win_coeffs, M, P, W2):
    # Keep in mind: M number of Taps, P Number of Channels, W2 Number of Spectra.
    # W2 was added to the original script to simplify some actions.
    W = x.shape[0]/ M / P
    # The original input is reshaped to a matrix with (nChannels x (nSpectra + nTaps -1))
    x_p = (x.reshape(((W2 + (M - 1)), P)).T).astype(float)
    # The original coefficients are reshaped to a matrix with (nChannels x nTaps)
    h_p = win_coeffs.reshape((M, P)).T
    # Create an empty array for the output with size (nChannels x nSpectra)
    x_summed = np.zeros((P, W2))
    # Foreach value in the input take nTaps predecessors of the same channel and build the new value by building the sum
    # of the multiplication of the predecessor with the window coefficients.
    for t in range(0, W2):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    # Bring the result into readable form, for writing into file for manual checks.
    transposed = x_summed.T
    transposed2 = np.around(transposed, decimals=6)
    oneline = np.reshape(transposed2, (x.shape[0]-P*(M-1),1))
    np.savetxt("/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/Python/" + 'firoutput.txt', oneline, fmt='%.6f')
    # Return the transposed matrix for future calculations.
    return x_summed.T


def fft(x_p, P, axis=1):
    # Not a lot to say here; check the official documentation for questions:
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.fft.rfft.html
    return np.fft.rfft(x_p, P, axis=1)


def pfb_filterbank(x, win_coeffs, M, P, W):
    # Calculate the FIR Filter
    x_fir = pfb_fir_frontend(x, win_coeffs, M, P, W)
    # Apply the FFT algorithm.
    x_pfb = fft(x_fir, P)
    # Two ways of writing the result, one with newline for each value one without newlines.
    # Keep in mind that we have imaginary numbers here.
    path = "/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/Python/" + 'output_unformated.txt'
    with open(path, 'w') as outfile:
        for data_slice in x_pfb:
            np.savetxt(outfile, data_slice, fmt='%.6f')
    np.savetxt("/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/Python/" + 'savetxtoutput.txt', x_pfb, fmt='%.6f')


if __name__ == "__main__":
    import pylab as plt
    import seaborn as sns
    sns.set_style("white")
    systemarguments = str(sys.argv)
    # Creates the directory for the Python output
    try:
        if not os.path.isdir("/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/Python"):
            os.mkdir("/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/Python")
    except OSError:
        print ("Creation of the directory failed")
    else:
        print ("Successfully created the directory ")
    M = int(sys.argv[2])       # Numbr of taps
    P = int(sys.argv[3])       # Number of 'branches', also fft length
    W = int(sys.argv[4])       # Number of windows of length M*P in input time stream

    # We assume input data exist from the CUDA script and read it.
    f = open("/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/input", 'r')
    if f.mode == "r":
        data = np.loadtxt(f)
    # We assume coefficients exist from the CUDA script and read it.
    f = open("/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/coeff", 'r')
    if f.mode == "r":
        win_coeffs = np.loadtxt(f)
    # Make a savety write to be able to ensure later all data is read correctly.
    # (especially relevant for changing data size).
    np.savetxt("/mnt/c/Users/b98/PFB_OUTPUT/" + sys.argv[1] + "/Python/" + 'data.txt', data, fmt='%.6f')
    np.set_printoptions(precision=3)
    # Start the calculations for the filterbank
    pfb_filterbank(data, win_coeffs, M, P, W)

