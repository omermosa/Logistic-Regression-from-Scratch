# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 04:46:06 2020

@author: OmerMoussa
"""

import math as mt
import numpy as np
import pandas as pd

#data are made into CSV.
DS=pd.read_csv("P2_D.csv")

#turn labels into 0 for neg, 1 for pos

DS["Class"]=DS["Class"].replace("+",1)
DS["Class"]=DS["Class"].replace("-",0)

#Drop Categorical Features text since they onehot encoding is enough
DS.drop(["Color","Tires","Type"],axis=1,inplace=True)

# traning and test split
X_train=DS.iloc[:11,:-1]
y_train=DS.iloc[:11,-1]
X_test=DS.iloc[11:,:-1]
y_test=DS.iloc[11:,-1]

X=DS.iloc[:,:-1]
y=DS.iloc[:,-1]
# logistic Reg function
def logReg(X,w,w0):
    return (1/(1+mt.exp(-1*np.dot(X,w)+w0)))



#build the model
 
def Err(j,w,w0,X_train,y_train):
        return logReg(np.array(X_train[j]),w,w0)-y_train[j]

#fit the model using Gradient Descent

def fit_GD(X_train,y_train,w0,w,dw0,dw):
        #max iterations
    n_it=1000
    eta =0.5
    #initialize weights
   
    m=len(X_train) #size of trainset
    #error functions
    
    for i in range(n_it):
        for j in range(m):
            # calculate the derivative for each training instance
            err=Err(j,w,w0,X_train,y_train)
            dw0=dw0+(err)
            for k in range(len(w)):
                dw[k]=dw[k]+(err)*X_train[j][k] # dericative of the logReg loss function (sum(yi.xi)-sum(y'i.xi))
        dw=dw/m #avg loss 
        dw0=dw0/m
        for k in range(len(w)):
            w[k]=w[k]-eta*dw[k]
        w0=w0-eta*dw0
    #return (w0,w)
        

def Predict(Xi,w,w0):
    seg=logReg(Xi,w,w0)
    #print(seg)
    if seg >0.5: #decision bound
        return 1
    else:
        return 0
    
def get_yPred(X_test,w0,w,ytest):
    # testing on validation 
    y_predicted=[]
    
    for i in range(len(X_test)):
        Xi=np.array(X_test[i])
        #print(Xi)
        y_predicted.append(Predict(Xi,w,w0))

    print("predicted: ", y_predicted, "Actual: ", np.array(ytest))
    return y_predicted

# building confusion matrix
def buildConfMat(y_predicted,y_test):
    TP,TN,FP,FN=0,0,0,0
   
    yt=np.array(y_test)
    for i in range (len(y_predicted)):
        if(y_predicted[i]==1):
            if(yt[i]==1):
                TP+=1
            else:
                FP+=1
        if(y_predicted[i]==0):
            if(yt[i]==0):
                TN+=1
            else:
                FN+=1    
    
    conf_mat=[[TP,FN],[FP,TN]]        
    print ("Conf Mat ", conf_mat)
    
    Pr=0
    Re=0
    if TP+FP !=0: #avoid undefined values e.g 0/0
        Pr=TP/(TP+FP)
    if TP+FN !=0:
        Re=TP/(TP+FN)
    print("Precision: ", Pr, "Recall: ", Re)
    Ac=(TP+TN)/len(y_test)
    return( Pr,Re,Ac)
    
    
def Kfold (k,DS ):
    data=DS.values
    Groups=[]
    L=len(DS)
    gsize=mt.ceil(L/k)
    w=[0]*(len(DS.columns)-1) #num of features
    w0=0
    dw0=0
    dw=np.array([0]*len(w))
    # partion into k groups
    for i in range (k):
        s=(i)*gsize
        e=min((i+1)*gsize,L)
        Groups.append(data[s:e])
    total_racall,total_precision,total_accurcy=0,0,0
    weights=[]
    Gcopy=Groups.copy()
    for i in range(k):
        Gcopy=Groups.copy()
        X=Groups[i]
        Xtest,ytest=X[:,:-1],X[:,-1]
        #print(Xtest)
        del(Gcopy[i])
        Gcopy=np.array(Gcopy)
        #print(Gcopy)
         #Gcopy=Gcopy.reshape(Gcopy.shape[0]*Gcopy.shape[1],Gcopy.shape[2])
        #construct the training sets
        Xt,yt=[],[]
        for l in Gcopy:
            for g in l:
                Xt.append(g[:-1])
                yt.append(g[-1])
            
        fit_GD(Xt,yt,w0,w,dw0,dw)
        #predict
        yp=get_yPred(Xtest,w0,w,ytest)
        #get the performance measures
        p,r,Ac=buildConfMat(yp,ytest)
        total_precision+=p
        total_racall+=r
        total_accurcy+=Ac
        weights.append(w) #get avg weights
        #print(weights)
    print("avg precision %s, avg recall %s, Avg Accuracy %s" %(total_precision/k,total_racall/k,total_accurcy/k))
        



Kfold(5,DS)
    




