#!/usr/bin/python  
#-*- coding:utf-8 -*-  
############################  
#File Name: kpls.py
#Author: Wenjie Zhang 
#Mail: zwjhit@gmail.com  
#Created Time: 2017-08-05 11:31:24
############################  

from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
from scipy.linalg import eigh
from sklearn import preprocessing


class kpls():

    def __init__(self, n_components=2, sigma=0.1, scale=True):
        self.n_components = n_components
        self.sigma=sigma
        self.scale = scale

    def fit(self, X, Y):
        if self.scale is True:
            self.scaler = preprocessing.StandardScaler().fit(X)
            self.X=scaler.transform(X)
        else:
            self.X=X
        K=rbf_kernel(X,X,self.sigma)
        T=np.zeros((X.shape[0],self.n_components))
        iter_count = 0
        while iter_count < self.n_components:
            L=np.dot(Y,Y.T)
            M=np.dot(K,L)
            eig_sub=M.shape[0]-1
            v,t=eigh(M,eigvals=(eig_sub,eig_sub))
            T[0:,iter_count]=t[0:,0]

            K_i= np.identity(X.shape[0])-np.dot(t,t.T)
            K_r= np.dot(np.dot(K_i,K),K_i)
            K=K_r
            Y=Y-np.dot(np.dot(t,t.T),Y)

            iter_count=iter_count+1
        self.T=T

    def fit_tranform(self,X,Y):
        fit(self,X,Y)
        return self.T

    def transform(self,Z):
        if self.scale is True:
            Z=self.scaler.transform(Z)
        K_zx=rbf_kernel(Z,self.X,self.sigma)
        return np.dot(K_zx,self.T)

