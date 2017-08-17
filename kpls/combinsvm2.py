#!/usr/bin/py://tex.stackexchange.com/questions/138976/change-line-spacing-in-algorithm2ethon  
#-*- coding:utf-8 -*-  
############################  
#File Name: kksvm.py
#Author: Wenjie Zhang 
#Mail: zwjhit@gmail.com  
#Created Time: 2017-08-10 09:19:23
############################  

from sklearn import preprocessing
import scipy.io as scio
import numpy as np
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from kpls2 import kpls
from openpyxl.reader.excel import load_workbook
import math
import pickle

def score1(label,data):
    precision=[]
    for i in range(4):
        if np.isnan(sum(data[:,i]*label[:,i])/sum(data[:,i])):
            precision.append(0)
        else:
            precision.append(sum(data[:,i]*label[:,i])/sum(data[:,i]))
    artifact_sensitivity=sum(data[:,3]*label[:,3])/sum(label[:,3])
    artifact_specificity=sum(sum(data[:,0:3]*label[:,0:3]))/36
    heartproblem_detection_sensitivity=sum(sum(data[:,1:3]*label[:,1:3]))/22
    heartproblem_detection_precision=sum(sum(data[:,1:3]*label[:,1:3]))/sum(sum(data[:,1:3]))
    y_index=sum(data[:,3]*label[:,3])/16-(1-artifact_specificity)
    f_score=1.81*heartproblem_detection_sensitivity*heartproblem_detection_precision/(
    0.81*heartproblem_detection_precision+heartproblem_detection_sensitivity)
    total_score=sum(precision)
    normal_score=(precision[0]*14+precision[1]*14+precision[2]*8+precision[3]*16)/52

    score_result=[]
    score_result.append({'Precision of Normal':precision[0]})
    score_result.append({'Precision of Murmur':precision[1]})
    score_result.append({'Precision of Extra Sound':precision[2]})
    score_result.append({'Precision of Artifact':precision[3]})
    score_result.append({'Artifact Sensitivity':artifact_sensitivity})
    score_result.append({'Artifact Specificity':artifact_specificity})
    score_result.append({'Heartproblem Detection Sensitivity':heartproblem_detection_sensitivity})
    score_result.append({'Heartproblem Detection Precision':heartproblem_detection_precision})
    score_result.append({'Youden Index of Artifact':y_index})
    score_result.append({'F-Score of Heartproblem Detection':f_score})
    score_result.append({'Total Precision':total_score})
    score_result.append({'Normalized Precision':normal_score})

    return score_result

def score2(label,data):
    precision=[]
    for i in range(3):
        if np.isnan(sum(data[:,i]*label[:,i])/sum(data[:,i])):
            precision.append(0)
        else:
            precision.append(sum(data[:,i]*label[:,i])/sum(data[:,i]))
    sensitivity_of_heartproblem=sum(sum(data[:,1:3]*label[:,1:3]))/59
    specificity_of_heartproblem=sum(data[:,0]*label[:,0])/136
    y_index=sensitivity_of_heartproblem+specificity_of_heartproblem-1
    dp=(np.sqrt(3)/np.pi)*(np.log10(sensitivity_of_heartproblem/(1-sensitivity_of_heartproblem))
        +np.log10(specificity_of_heartproblem/(1-specificity_of_heartproblem)))
    total_score=sum(precision)
    normal_score=(precision[0]*136+precision[1]*39+precision[2]*20)/195

    score_result=[]
    score_result.append({'Precision of Normal':precision[0]})
    score_result.append({'Precision of Murmur':precision[1]})
    score_result.append({'Precision of Extrastole':precision[2]})
    score_result.append({'Sensitivity of Heartproblem':sensitivity_of_heartproblem})
    score_result.append({'Specificity of Heartproblem':specificity_of_heartproblem})
    score_result.append({'Youden Index of Artifact':y_index})
    score_result.append({'Discriminant Power':dp})
    score_result.append({'Total Precision':total_score})
    score_result.append({'Normalized Precision':normal_score})
    
    return score_result

def getdata():
    data=scio.loadmat('sdata.mat')
    x1=data['x1']#31
    x2=data['x2']#34
    x3=data['x3']#19
    x4=data['x4']#40
    tx=data['tx']
    data=np.concatenate((x1,x2))
    data=np.concatenate((data,x3))
    data=np.concatenate((data,x4))
    target=np.ones((data.shape[0],))
    target[31:31+34]=2
    target[31+34:31+34+19]=3
    target[31+34+19:34+31+19+40]=4
    
    Y=np.zeros((data.shape[0],4))
    Y[0:31,0]=1
    Y[31:31+34,1]=1
    Y[31+34:31+34+19,2]=1
    Y[31+34+19:34+31+19+40,3]=1

    return data,target,Y,tx


def get_kpls_feature(data,Y,tx,n_components,scale1,scale2,sigma):
    
    kp=kpls(n_components=n_components,scale1=scale1,scale2=scale2,sigma=sigma)
    newdata=kp.fit_transform(data,Y)
    newtx=kp.transform(tx)

    return newdata,newtx

def get_svm_score(newdata,target,newtx,gamma,C):
    clf=svm.SVC(kernel='rbf',gamma=gamma,C=C,class_weight={1:1,2:1,3:1,4:1})
    clf.fit(newdata,target)
    print(clf)
    pre_results=clf.predict(newtx)
    results=np.zeros((52,4))
    for i in range(52):
        results[i][int(pre_results[i])-1]=1
    #print(labels)
    f=open('label.dat','rb')
    label=pickle.load(f)
    f.close()
    cc=score1(label[0],results)
    return cc

def tunepara(data,test,label,gamma,C):
    scores=np.zeros((len(gamma),len(C)))
    for idx,vali in enumerate(gamma):
        for jdx,valj in enumerate(C):
            score_result=get_svm_score(data,label,test,vali,valj)
            scores[idx,jdx]=score_result[11]['Normalized Precision']
    return scores

def tunekernel(data,test,label,target,kernel,gamma,C):
    scores=np.zeros((len(kernel),))
    n_components=50
    scale1=1
    scale2=1
    for idx,val in enumerate(kernel):
        newdata,newtx=get_kpls_feature(data,label,test,n_components,scale1,scale2,val)
        score_result=get_svm_score(newdata,target,newtx,gamma,C)
        scores[idx]=score_result[11]['Normalized Precision']

    return scores
if  __name__=="__main__":
    n_components=50
    scale1=1
    scale2=1
    sigma=0.08
    C=150
    gamma='auto'
    data,target,Y,tx=getdata()
    newdata,newtx=get_kpls_feature(data,Y,tx,n_components,scale1,scale2,sigma)
    score_result=get_svm_score(newdata,target,newtx,gamma,C)

    #for i in score_result:
    #    print(i)
    
    #kernel=[]
    #for i in range(100):
    #    kernel.append(0.001*(i+1))

    #gamma='auto'
    #C=150
    #label=Y
    #test=tx
    #scores=tunekernel(data,test,label,target,kernel,gamma,C)
    #for i,j in enumerate(scores):
    #    print(i,j)
    ##



    pls2=PLSRegression(n_components=50,scale=1)
    newdata2=pls2.fit_transform(data,Y)[0]
    newtx2=pls2.transform(tx)
#    print(max(newdata.max(axis=0)))
    
    combine_data=np.concatenate((newdata*9,newdata2),axis=1)
    combine_tx=np.concatenate((newtx*9,newtx2),axis=1)

    
    C=150
    gamma='auto'

    score_result=get_svm_score(combine_data,target,combine_tx,gamma,C)
#    score_result=get_svm_score(newdata,target,newtx,gamma,C)
    #score_result=get_svm_score(newdata2,target,newtx2,gamma,C)
    for i in score_result:
        print (i)
