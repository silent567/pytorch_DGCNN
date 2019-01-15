#!/usr/bin/env python
# coding=utf-8

import numpy as np
from pygfl.easy import solve_gfl

def sparsemax(z,gamma=1):
    z = z / gamma
    z_sorted = np.sort(z)[::-1]
    cumsum = np.cumsum(z_sorted)
    k = 0
    while k<z.size and 1+(k+1)*z_sorted[k] > cumsum[k]:
        k += 1
    tau = (cumsum[k-1]-1)/(k)
    return np.maximum(z-tau,0)

def softmax(z):
    z_exp = np.exp(z)
    return z_exp/np.sum(z_exp)

def gfusedlasso_with_edge(z,edge,lam=None):
    z_fused = solve_gfl(z.astype(np.float64),edge,lam=lam)
    return z_fused.astype(z.dtype)

def gfusedlasso(z,A,lam=None):
    # print(type(z),type(A),type(lam))
    A = np.triu(A) > 0
    edges = np.stack(np.mask_indices(A.shape[0],lambda n,k:A),axis=-1)
    # print(z.shape,z.dtype,edges.shape,edges.dtype,lam)
    z_fused = solve_gfl(z.astype(np.float64),edges,lam=lam)
    return z_fused.astype(z.dtype)

def gfusedmax(z,A,lam=None,gamma=1):
    z_fused = gfusedlasso(z,A,lam)
    return sparsemax(z_fused,gamma)


