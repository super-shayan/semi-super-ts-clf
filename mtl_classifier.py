#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:02:46 2019

@author: shayan
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.metrics import mean_squared_error
import copy
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, default="WordSynonyms")
parser.add_argument("--horizon", type=float, default=0.2)
parser.add_argument("--stride", type=float, default=0.2)
parser.add_argument("--seed1", type=int, default=17)
parser.add_argument("--seed2", type=int, default=10)
parser.add_argument("--alpha", type=float, default=0.1)

args = parser.parse_args()
filename=args.filename
horizon=args.horizon
stride=args.stride
seed1=args.seed1
seed2=args.seed2
alpha=args.alpha
#sys.stdout=open("clf_mtl_"+str(seed1)+"_"+str(seed2)+"_"+filename+"_"+str(horizon)+"_"+str(stride)+"_"+str(alpha)+".log","w")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class MTNet(torch.nn.Module): 
    def __init__(self, horizon):
        super(MTNet, self).__init__()
        self.conv1 = nn.Conv1d(x_train.shape[1], 128, 9, padding=(9 // 2))
        self.bnorm1 = nn.BatchNorm1d(128)        
        self.conv2 = nn.Conv1d(128, 256, 5, padding=(5 // 2))
        self.bnorm2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 3, padding=(3 // 2))
        self.bnorm3 = nn.BatchNorm1d(128)        
        self.classification_head = nn.Linear(128, nb_classes)
        self.forecasting_head = nn.Linear(128, horizon)
        
    def forward(self, x_class, x_forecast):
        b1_c = F.relu(self.bnorm1(self.conv1(x_class)))
        b1_f = F.relu(self.bnorm1(self.conv1(x_forecast)))
        
        b2_c = F.relu(self.bnorm2(self.conv2(b1_c)))
        b2_f = F.relu(self.bnorm2(self.conv2(b1_f)))
        
        b3_c = F.relu(self.bnorm3(self.conv3(b2_c)))
        b3_f = F.relu(self.bnorm3(self.conv3(b2_f)))
        
        classification_features = torch.mean(b3_c, 2)#(64,128)#that is now we have global avg pooling, 1 feature from each conv channel
        forecasting_features = torch.mean(b3_f, 2)
        
        classification_out=self.classification_head(classification_features)
        forecasting_out=self.forecasting_head(forecasting_features)
        return classification_out, forecasting_out

    def forward_test(self, x_class):
        b1_c = F.relu(self.bnorm1(self.conv1(x_class)))
        b2_c = F.relu(self.bnorm2(self.conv2(b1_c)))
        b3_c = F.relu(self.bnorm3(self.conv3(b2_c)))
        classification_features = torch.mean(b3_c, 2)#(64,128)#that is now we have global avg pooling, 1 feature from each conv channel
        classification_out=self.classification_head(classification_features)
        return classification_out
        
def optimize_network(x_batch_class, y_batch_class, x_forecast, y_forecast):
    y_hat_classification, y_hat_forecasting = mtnet(x_batch_class.float(), x_forecast.float())

    loss_classification = criterion_classification(y_hat_classification, y_batch_class.long())
    loss_forecasting = criterion_forecasting(y_hat_forecasting, y_forecast.float())
    
    loss_mtl = loss_classification+alpha*loss_forecasting    
    optimizer.zero_grad()
    loss_mtl.backward()
    optimizer.step()
    return loss_classification.item(), loss_forecasting.item()

train =pd.read_csv("/run/user/1001/gvfs/sftp:host=mirror.ismll.de//home/shayan/ts/UCRArchive_2018/"+filename+"/"+filename+"_TRAIN.tsv",sep="\t",header=None)
test=pd.read_csv("/run/user/1001/gvfs/sftp:host=mirror.ismll.de//home/shayan/ts/UCRArchive_2018/"+filename+"/"+filename+"_TEST.tsv",sep="\t",header=None)
     
df = pd.concat((train,test))
y_s = df.values[:,0]
nb_classes = len(np.unique(y_s))
y_s = (y_s - y_s.min())/(y_s.max()-y_s.min())*(nb_classes-1)
df[df.columns[0]] = y_s

train, test = train_test_split(df, test_size=0.2, random_state=seed1)
train_labeled, train_unlabeled = train_test_split(train, test_size=1-0.1, random_state=seed2)            
train_unlabeled[train_unlabeled.columns[0]]=-1#Explicitly set all the instance's labels to -1

train_1=pd.concat((train_labeled,train_unlabeled))
x_train=train_1.values[:,1:]
y_train=train_1.values[:,0]

x_test=test.values[:,1:]
y_test=test.values[:,0]

x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean)/(x_train_std)
x_test = (x_test - x_train_mean)/(x_train_std)

x_train=x_train[:,np.newaxis,:] 
x_test=x_test[:,np.newaxis,:]

#x_train=x_train[:,:,np.newaxis] 
#x_test=x_test[:,:,np.newaxis]
max_acc_possible = 1-(sum([list(y_test).count(x) for x in list(set(np.unique(y_test))-set(np.unique(y_train)))])/len(y_test))

x_train = torch.from_numpy(x_train).to(device)
y_train = torch.from_numpy(y_train).to(device)
x_test = torch.from_numpy(x_test).to(device)
y_test = torch.from_numpy(y_test).to(device)

mtnet = MTNet(int(x_train.shape[2]*horizon)).to(device)

criterion_classification = nn.CrossEntropyLoss()
criterion_forecasting = nn.MSELoss()
optimizer = torch.optim.Adam(mtnet.parameters(), lr=1e-4)    

batch_size = 32
accuracies=[]

def return_sliding_windows(X):
    xf=[]
    yf=[]
    for i in range(0,X.shape[2],int(stride*X.shape[2])):
        horizon1=int(horizon*X.shape[2])
        if(i+horizon1+horizon1<=X.shape[2]):
#            print("X===>",i,i+horizon)
#            print("Y===>",i+horizon,i+horizon+horizon)
            xf.append(X[:,:,i:i+horizon1])
            yf.append(X[:,:,i+horizon1:i+horizon1+horizon1])
    
    xf=torch.cat(xf)
    yf=torch.cat(yf)
    
    return xf,yf

x_sliding_window, y_sliding_window = return_sliding_windows(x_train)

def shuffler(x_train, y_train):
    indexes=np.array(list(range(x_train.shape[0])))
    np.random.shuffle(indexes)
    x_train=x_train[indexes]
    y_train=y_train[indexes]
    return x_train, y_train
   
for t in range(5000):
    losses=[]
    x_train, y_train = shuffler(x_train, y_train)
    x_train_batch=x_train[y_train!=-1]
    y_train_batch=y_train[y_train!=-1]
    
    x_sliding_window, y_sliding_window = shuffler(x_sliding_window, y_sliding_window)
    
    for i in range(0,x_sliding_window.shape[0],batch_size):
        if i+batch_size<=x_sliding_window.shape[0]:
            closs,floss = optimize_network(x_train_batch, y_train_batch, x_sliding_window[i:i+batch_size], y_sliding_window[i:i+batch_size])
            losses.append([closs,floss])
        else:
            closs,floss = optimize_network(x_train_batch, y_train_batch, x_sliding_window[i:], y_sliding_window[i:])
            losses.append([closs,floss])
            
    val_acc = accuracy_score(np.argmax(mtnet.forward_test(x_test.float()).cpu().detach().numpy(),1),y_test.long().cpu().numpy())
    accuracies.append(val_acc)
    print("Epoch: ",t,"| Accuracy: ",val_acc, "/",max(accuracies),"/",max_acc_possible, "| Avg. losses: ", np.mean([loss[0] for loss in losses]),np.mean([loss[1] for loss in losses]), flush=True)
    if val_acc==1.0:
        break;
