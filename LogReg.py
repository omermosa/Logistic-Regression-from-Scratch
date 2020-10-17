# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 04:46:06 2020

@author: OmerHassan
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

# logistic Reg function
def logReg(X,w,w0):
    return (1/(1+mt.exp(-1*np.dot(X,w)+w0)))
    
#max iterations
n_it=1000
eta =0.4
#initialize weights
w=[0]*(len(DS.columns)-1) #num of features
w0=0
dw0=0
dw=np.array([0]*len(w))
m=len(X_train)
#error functions
def Err(j):
    return logReg(np.array(X_train.iloc[j,:]),w,w0)-y_train[j]

for i in range(n_it):
    for j in range(m):
        # calculate the derivative for each training instance
        dw0=dw0+(Err(j))
        for k in range(len(w)):
            dw[k]=dw[k]+(Err(j))*X_train.iloc[j,k] # dericative of the logReg loss function (sum(yi.xi)-sum(y'i.xi))
    dw=dw/m #avg loss 
    dw0=dw0/m
    for k in range(len(w)):
        w[k]=w[k]-eta*dw[k]
    w0=w0-eta*dw0

def Predict(Xi,w,w0):
    seg=logReg(Xi,w,w0)
    #print(seg)
    if seg >0.5: #decision bound
        return 1
    else:
        return 0
    
   
# testing on validation 
y_predicted=[]

for i in range(len(X_test)):
    Xi=np.array(X_test.iloc[i,:])
    #print(Xi)
    y_predicted.append(Predict(Xi,w,w0))

print("predicted: ", y_predicted, "Actual: ", np.array(y_test))

# building confusion matrix
TP=0
TN=0
FP=0
FN=0
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
print("Precision: ", TP/(TP+FP), "Recall: ", TP/(TP+FN))


