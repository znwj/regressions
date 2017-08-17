#!/usr/bin/python  
#-*- coding:utf-8 -*-  
############################  
#File Name: kpls.py
#Author: Wenjie Zhang 
#Mail: zwjhit@gmail.com  
#Created Time: 2017-08-05 11:31:24
#Reference paper:Rosipal R, Trejo L J. Kernel partial least squares regression in reproducing kernel hilbert space[J]. Journal of machine learning research, 2001, 2(Dec): 97-123.
#Kernel-NIPALS-PLS
############################  

from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from scipy.linalg import eigh,norm
from sklearn import preprocessing

def _center_scale_xy(X, Y, scale=True):
    """ Center X, Y and scale if the scale parameter==True
    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std

def _center_scale_kernel_matrix(K):
    n=K.shape[0]
    one_matrix=(1/n)*np.ones((n,n))
    K_center=K-2*np.dot(one_matrix,K)+np.dot(np.dot(one_matrix,K),one_matrix)
    return K_center

class kpls():

    def __init__(self, n_components=2, sigma=0.1, scale1=True, scale2=True):
        self.n_components = n_components
        self.sigma=sigma
        self.scale1 = scale1
        self.scale2 = scale2

    def fit(self, X, Y):
        self.X, self.Y, self.x_mean_, self.y_mean_, self.x_std_, self.y_std_ = (
            _center_scale_xy(X, Y, self.scale1))
        K=rbf_kernel(self.X,self.X,self.sigma)
        if self.scale2:
            XK=_center_scale_kernel_matrix(K)
        else:
            XK=K
        YK=self.Y
        T=np.zeros((X.shape[0],self.n_components))
        XK_norm=np.zeros((self.n_components,))
        YK_norm=np.zeros((self.n_components,))
        for iter_count in range(self.n_components):
            L=np.dot(YK,YK.T)
            M=np.dot(XK,L)
            eig_sub=M.shape[0]-1
            v,t=eigh(M,eigvals=(eig_sub,eig_sub))

            T[0:,iter_count]=t[0:,0]

            K_i= np.identity(X.shape[0])-np.dot(t,t.T)
            K_r= np.dot(np.dot(K_i,K),K_i)
            XK=K_r
            YK=YK-np.dot(np.dot(t,t.T),YK)
            XK_norm[iter_count]=norm(XK)
            YK_norm[iter_count]=norm(YK)
            

        self.T=T
        self.XK_norm=XK_norm
        self.YK_norm=YK_norm

    def fit_transform(self,X,Y):
        self.fit(X,Y)
        return self.T

    def transform(self,Z):
        Z-= self.x_mean_
        Z /= self.x_std_
        K_zx=rbf_kernel(Z,self.X,self.sigma)
        return np.dot(K_zx,self.T)

