#!/usr/bin/python  
#-*- coding:utf-8 -*-  
############################  
#File Name: example.py
#Author: Wenjie Zhang 
#Mail: zwjhit@gmail.com  
#Created Time: 2017-08-05 11:19:35
############################  


from sklearn.datasets import load_digits
import numpy as np
from kpls import kpls

#change target to lable matrix, target range 0--integer, example 0--9
def to_label(target,mint,maxt):
    label_target=np.zeros((target.shape[0],maxt-mint+1))
    for i in range(target.shape[0]):
        label_target[i-1,target[i-1]]=1
    return label_target

if __name__=="__main__":
    digits = load_digits()
    data=digits['data']
    target=digits['target']
    target=to_label(target,0,9)

    cc=kpls()
    cc.fit(data,target)
    
