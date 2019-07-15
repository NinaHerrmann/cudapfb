import numpy as np


def pfb_fir_frontend(x, win_coeffs, M, P):
    W = x.shape[0] / M / P
    x_p = x.reshape((W*M, P)).T
    h_p = win_coeffs.reshape((M, P)).T
    x_summed = np.zeros((P, M * W - M))
    for t in range(0, M*W-M):
        x_weighted = x_p[:, t:t+M] * h_p
        x_summed[:, t] = x_weighted.sum(axis=1)
    return x_summed.T


if __name__ == '__main__':

    x = np.array([0.1, 0.2, 0.22, 0.4444, 0.55555, 0.66666])
    win_coeffs = np.array([0.5,0.2])
    M = 2
    P = 3
    pfb_fir_frontend(x, win_coeffs, M, P)
