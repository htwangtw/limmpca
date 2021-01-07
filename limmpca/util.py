import pandas as pd
import numpy as np


def varimax(Phi, gamma=1, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        rot_tmp = np.diag(np.diag(np.dot(Lambda.T,Lambda)))
        lam_rotate = np.dot(Lambda, rot_tmp)
        mat = np.asarray(Lambda)**3 - (gamma / p) * lam_rotate
        mat = np.dot(Phi.T, tmp)
        u, s, vh = np.linalg.svd(mat)
        R = np.dot(u, vh)
        d = np.sum(s)
        if d/d_old < tol: break
    return np.dot(Phi, R)


def rescale(data):
    """
    Rescale the data to ensure the smallest value is always 0 and max 1
    """
    data = data.astype(float)
    ceiling = np.max(data.ravel())
    floor = np.min(data.ravel())
    return (data - floor) / (ceiling - floor)
