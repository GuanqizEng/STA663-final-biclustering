#!/usr/bin/env python
# coding: utf-8

import scipy.linalg as la
import numpy as np
from sparsesvd import sparsesvd 
from scipy.sparse import csc_matrix
import numba
from numba import jit
import warnings
warnings.filterwarnings('ignore')

@jit (nopython = True, parallel = True)

def BICv_numba(X, u, v_tilde, sigma2_hat_v, n, d): 
    """
    This function return the n*d times of the original BIC.
    Only used for internal use for the SSVD function.
    It's used when updating v.
    """
    df = np.sum(np.abs(v_tilde) > 1e-06)
    bic_nd = np.sum((X - u @ v_tilde.T)**2) / (sigma2_hat_v) + np.log(n*d) * df
    #bic_nd = np.sum((X - np.outer(u, v_tilde))**2) / (sigma2_hat_v) + np.log(n*d) * df
    #bic_nd = np.trace((X - uvt) @ (X - uvt).T) / (sigma2_hat_v) + np.log(n*d) * df
    return bic_nd

def BICu_numba(X, v, u_tilde, sigma2_hat_u, n, d):
    """
    This function return the n*d times of the original BIC.
    Only used for internal use for the SSVD function.
    It's used when updating u.
    """
    df = np.sum(np.abs(u_tilde) > 1e-06)
    bic_nd = np.sum((X - u_tilde @ v.T)**2) / (sigma2_hat_u) + np.log(n*d) * df
    #bic_nd = np.sum((X - np.outer(u_tilde, v))**2) / (sigma2_hat_u) + np.log(n*d) * df
    #bic_nd = np.trace((X - uvt) @ (X - uvt).T) / (sigma2_hat_u) + np.log(n*d) * df
    return bic_nd

def opt_lambda_v_numba(X, lamd_grid, Xu_nonzero, w2_nonzero, u, sigma2_hat_v, n, d, index_v):
    """
    This function return best lambda that minimize the correponding BIC.
    Only used for internal use for the SSVD function.
    It's used when updating v.
    """
    BICs = np.ones(lamd_grid.shape[0])

    for i in numba.prange(BICs.shape[0]):
        v_tilde_nonzero = np.sign(Xu_nonzero) * (np.abs(Xu_nonzero) >= lamd_grid[i]*w2_nonzero/2) * (np.abs(Xu_nonzero) - lamd_grid[i]*w2_nonzero/2)
        v_tilde = np.zeros((d,1))
        v_tilde[index_v] = v_tilde_nonzero
        BICs[i] = BICv_numba(X, u, v_tilde, sigma2_hat_v, n, d)
    lamd_min = np.argmin(BICs)
    return lamd_grid[lamd_min]

def opt_lambda_u_numba(X, lamd_grid, Xv_nonzero, w1_nonzero, v, sigma2_hat_u, n, d, index_u):
    """
    This function return best lambda that minimize the correponding BIC.
    Only used for internal use for the SSVD function.
    It's used when updating u.
    """
    BICs = np.ones(lamd_grid.shape[0])
    
    for i in numba.prange(BICs.shape[0]):
        u_tilde_nonzero = np.sign(Xv_nonzero) * (np.abs(Xv_nonzero) >= lamd_grid[i]*w1_nonzero/2) * (np.abs(Xv_nonzero) - lamd_grid[i]*w1_nonzero/2)
        u_tilde = np.zeros((n,1))
        u_tilde[index_u] = u_tilde_nonzero
        BICs[i] = BICu_numba(X, v, u_tilde, sigma2_hat_u, n, d)
    
    #BICs = list(map(lambda x: BIC(x, w, v, u_tilde, Z, sigma2_hat_u, n, d), lamd_grid))
    lamd_min = np.argmin(BICs)
    return lamd_grid[lamd_min]

def uv_renew_numba(u, s, v, X, gamma1, gamma2):
    """
    This function will return the updated u, v and the corresponding lambdas.
    Only for internal use for the SSVD function.
    """
    n,d = X.shape
    u = u.reshape((n,1))
    v = v.reshape((d,1))
    SSTO = np.sum(X**2)
        
    ## first, update v
    
    # compute the weights, which are OLS for v (Xu is also the ols)
    Xu = np.dot(X.T, u) # this is also the v_tilde in the paper, Xu is (d,1)
    w2 = np.abs(Xu)**(-gamma2)
    
    # compute the estimated sigma2 hat for v
    uvt = np.outer(u,v)
    #uvt = u @ v.T
    sigma2_hat_v = np.trace((X - s*uvt)@(X - s*uvt).T) / (n*d - d) 
    #sigma2_hat_v = np.abs(SSTO - sum(Xu**2)) / (n*d - d) 
    
    # then, find the possible lambdas for v
    # notice that, equivantly, we can write 2 * (X.T @ u) / w2 > lambda_v, and 2 * (X.T @ v) / w1 > lambda_u
    # thus, it only makes sense to search different lambdas according to the values of (X.T @ u)/w2 or (X.T @ v)/w1
    index_v = np.where(w2 < 1e8) # the index where Xu is non-zero. Out of these values, the v will almost be zero.
    index_v = index_v[0]
    Xu_nonzero = Xu[index_v]
    w2_nonzero = w2[index_v]
    lamd_grid_v = 2 * Xu_nonzero / w2_nonzero
    #lamd_grid_v =  Xu[index_v] / w2[index_v]
    lamd_grid_v = np.unique(np.append(0, np.abs(lamd_grid_v)))
    lamd_grid_v.sort()
    lamd_grid_v = lamd_grid_v[0:-1]
    lamd_grid_v = np.r_[lamd_grid_v, np.linspace(0, lamd_grid_v[-1], num = 50)]
    
    
    # find the optimized lambda for v
    lamd_v = opt_lambda_v_numba(X, lamd_grid_v, Xu_nonzero, w2_nonzero, u, sigma2_hat_v, n, d, index_v)
    
    # update v
    sig_v = np.sign(Xu)
    #v_new = sig_v * (np.abs(Xu) - lamd_v*w2/2) * (np.abs(Xu) >= lamd_v*w2/2) / la.norm(Xu)
    v_new = sig_v * (np.abs(Xu) - lamd_v*w2/2) * (np.abs(Xu) >= lamd_v*w2/2)
    v_new = v_new / la.norm(v_new)
    
    ## then, update the u
    
    # compute the weights for u
    Xvnew = np.dot(X, v_new) # this is also the u_tilde in the paper, Xvnew is (n,1)
    w1 = np.abs(Xvnew)**(-gamma1)
    
    # compute the estimated sigma2 hat for u
    uvt = np.outer(u,v_new)
    #uvt = u @ v_new.T
    sigma2_hat_u = np.trace((X - s*uvt)@(X - s*uvt).T) / (n*d - d)
    #sigma2_hat_u = np.abs(SSTO - sum(Xvnew**2)) / (n*d - n) 
    
    # then, find the possible lambdas for u
    index_u = np.where(w1 < 1e8)
    index_u = index_u[0]
    Xv_nonzero = Xvnew[index_u]
    w1_nonzero = w1[index_u]
    lamd_grid_u = 2 * Xv_nonzero / w1_nonzero
    lamd_grid_u = np.unique(np.append(0, np.abs(lamd_grid_u)))
    lamd_grid_u.sort()
    lamd_grid_u = lamd_grid_u[0:-1]
    lamd_grid_u = np.r_[lamd_grid_u, np.linspace(0, lamd_grid_u[-1], num = 50)]
    
    # find the optimized lambda for u
    lamd_u = opt_lambda_u_numba(X, lamd_grid_u, Xv_nonzero, w1_nonzero, v_new, sigma2_hat_u, n, d, index_u)
    
    # update u
    sig_u = np.sign(Xvnew)
    u_new = sig_u * (np.abs(Xvnew) - lamd_u*w1/2) * (np.abs(Xvnew) >= lamd_u*w1/2)
    u_new = u_new / la.norm(u_new)
    
    return v_new, u_new, lamd_v, lamd_u

def SSVD_numba(X, gamma1, gamma2, max_iter = 100, tol = 1e-05):
    """
    This function returns the rank 1 approximation for a sparse matrix.
    Recommended for use when you have large dataset, for example, with more than 10 thousands columns.
    Not recommend for very small data set.
    
    X: the input matrix
    gamma1: known power parameter for v
    gamma2: known power parameter for u
    max_iter: max iteration
    tol: tolerence. If the steps between old u and v and the new ones are less than tol, then it stops.
    
    return: (number of iter, u, v, s, lambda_u, lambda_v)
    """
    #first, get the stuffs in step 1
    ut, s, vt = sparsesvd(csc_matrix(X), k = 1) # the returned vectors are all with 1 row
    u_curr = ut.T 
    v_curr = vt.T
    n,d = X.shape
    
    # then, come to the step 2
    for i in range(max_iter):
        
        # update v
        v_new, u_new, lambda_v, lambda_u = uv_renew_numba(u_curr, s, v_curr, X, gamma1, gamma2)
        if la.norm((v_new - v_curr)) < tol and la.norm((u_new - u_curr)) < tol :
            return i+1, u_new, v_new, u_new.T @ X @ v_new, lambda_u, lambda_v
        else:
            u_curr = u_new
            v_curr = v_new
        
    print("Results haven't converged. Please increase the number of iterations.")
    return max_iter, u_curr, v_curr, u_curr.T @ X @ v_curr, lambda_u, lambda_v