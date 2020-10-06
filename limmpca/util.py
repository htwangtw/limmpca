import pandas as pd 
import numpy as np 


def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 \
        - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)


def correct_scale(data, labels):
    # correct each subject by used scale range
    for id in np.unique(data.RIDNO):
        id_idx = data['RIDNO'].str.match(id)
        cur = data[id_idx].loc[:, labels].values
        scling = np.max(cur.flatten())
        floor = np.min(cur.flatten())
        cur = (cur - floor) / scling
        # update
        data[id_idx].loc[:, labels] = cur
    return data