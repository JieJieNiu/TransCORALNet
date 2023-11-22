#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:11:12 2023

@author: jolie
"""

import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from torchsummary import summary
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import math
from torch import optim
from torch.nn import init
import os
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
from model import TransCORALNet


###Predict
###Load model
model = TransCORALNet()
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint)

def prediction(target_test_loader):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        results_test=[]
        correct= 0
        FN_test= 0
        TN_test= 0
        FP_test= 0
        TP_test= 0
        clf_criterion = nn.CrossEntropyLoss()
        for test_data, test_target in target_test_loader:
            test_data, test_target = Variable(test_data), Variable(test_target)
            target_test_preds,_= model(test_data,test_data)
            test_pred = target_test_preds.data.max(1)[1]
            correct += test_pred.eq(test_target.data.view_as(test_pred)).cpu().sum()
            test_loss += clf_criterion(target_test_preds, test_target.to(dtype=torch.long)).item()
            zes=Variable(torch.zeros(BATCH_SIZE).type(torch.LongTensor))#全0变量
            ons=Variable(torch.ones(BATCH_SIZE).type(torch.LongTensor))#全1变量
            test_correct01 = ((test_pred==zes)&(test_target==ons)).sum() #原标签为1，预测为 0 的总数
            test_correct10 = ((test_pred==ons)&(test_target==zes)).sum() #原标签为0，预测为1 的总数
            test_correct11 = ((test_pred==ons)&(test_target==ons)).sum()#原标签为1，预测为 1 的总数
            test_correct00 = ((test_pred==zes)&(test_target==zes)).sum()#原标签为0，预测为 0 的总数
            FN_test += test_correct01.item()
            FP_test += test_correct10.item()
            TP_test += test_correct11.item()
            TN_test += test_correct00.item()

    acc_test=100.*correct/len(target_test_loader.dataset)
    recall_test=TP_test/(TP_test+FN_test)
    percision_test=TP_test/(TP_test+FP_test)
    F1_test=2*(recall_test*percision_test)/(recall_test+percision_test)
    test_loss =test_loss/len(target_test_loader)
    return acc_test,recall_test,percision_test,F1_test,test_loss
    print("Accuracy:{}/tRecall:{}/tPrecision:{}/tF1:{}/tloss{}".format(acc_test,recall_test,percision_test,F1_test,test_loss))
    
####Different testing dataset
industry_cut=(80,60,40,30,20,10)


##Validation
validation_d=torch.tensor(np.array(validation_d)).to(dtype=torch.float32)
validation_l=torch.tensor(np.array(validation_l)).to(dtype=torch.float32)
dataset=TensorDataset(validation_d,validation_l)
validation_loader=DataLoader(dataset=dataset,batch_size=1)
for i, dataset in enumerate(validation_loader):
    data,label=dataset
prediction_val=prediction(validation_loader)
print(prediction_val)

###80
data_test_i=data_test[data_test['segement_industry'].isin(special_80[0:industry_cut[0]])]
target_test_d_80=np.array(data_test_i.iloc[:,1:-1])
target_test_l_80=np.array(data_test_i.iloc[:,-1])
target_test_d_80=torch.tensor(target_test_d_80).to(dtype=torch.float32)
target_test_l_80=torch.tensor(target_test_l_80).to(dtype=torch.float32)
dataset_80=TensorDataset(target_test_d_80,target_test_l_80)
target_test_loader_80=DataLoader(dataset=dataset_80,batch_size=1)
for i, dataset in enumerate(target_test_loader_80):
    data,label=dataset
prediction_80=prediction(target_test_loader_80)
prediction(target_test_loader_80)

###60
data_test_i=data_test[data_test['segement_industry'].isin(special_80[0:industry_cut[1]])]
target_test_d_60=np.array(data_test_i.iloc[:,1:-1])
target_test_l_60=np.array(data_test_i.iloc[:,-1])
target_test_d_60=torch.tensor(target_test_d_60).to(dtype=torch.float32)
target_test_l_60=torch.tensor(target_test_l_60).to(dtype=torch.float32)
dataset_60=TensorDataset(target_test_d_60,target_test_l_60)
target_test_loader_60=DataLoader(dataset=dataset_60,batch_size=1)
for i, dataset in enumerate(target_test_loader_60):
    data,label=dataset
prediction_60=prediction(target_test_loader_60)
prediction(target_test_loader_60)

###40
data_test_i=data_test[data_test['segement_industry'].isin(special_80[0:industry_cut[2]])]
target_test_d_40=np.array(data_test_i.iloc[:,1:-1])
target_test_l_40=np.array(data_test_i.iloc[:,-1])
target_test_d_40=torch.tensor(target_test_d_40).to(dtype=torch.float32)
target_test_l_40=torch.tensor(target_test_l_40).to(dtype=torch.float32)
dataset_40=TensorDataset(target_test_d_40,target_test_l_40)
target_test_loader_40=DataLoader(dataset=dataset_40,batch_size=1)
for i, dataset in enumerate(target_test_loader_40):
    data,label=dataset
prediction_40=prediction(target_test_loader_40)
prediction(target_test_loader_40)

###30
data_test_i=data_test[data_test['segement_industry'].isin(special_80[0:industry_cut[3]])]
target_test_d_30=np.array(data_test_i.iloc[:,1:-1])
target_test_l_30=np.array(data_test_i.iloc[:,-1])
target_test_d_30=torch.tensor(target_test_d_30).to(dtype=torch.float32)
target_test_l_30=torch.tensor(target_test_l_30).to(dtype=torch.float32)
dataset_30=TensorDataset(target_test_d_30,target_test_l_30)
target_test_loader_30=DataLoader(dataset=dataset_30,batch_size=1)
for i, dataset in enumerate(target_test_loader_30):
    data,label=dataset
prediction_30=prediction(target_test_loader_30)
prediction(target_test_loader_30)
    
###25
data_test_i=data_test[data_test['segement_industry'].isin(special_80[0:industry_cut[4]])]
target_test_d_20=np.array(data_test_i.iloc[:,1:-1])
target_test_l_20=np.array(data_test_i.iloc[:,-1])
target_test_d_20=torch.tensor(target_test_d_20).to(dtype=torch.float32)
target_test_l_20=torch.tensor(target_test_l_20).to(dtype=torch.float32)
dataset_20=TensorDataset(target_test_d_20,target_test_l_20)
target_test_loader_20=DataLoader(dataset=dataset_20,batch_size=1)
for i, dataset in enumerate(target_test_loader_20):
    data,label=dataset
prediction_20=prediction(target_test_loader_20)
prediction(target_test_loader_20)

        
###10
data_test_i=data_test[data_test['segement_industry'].isin(special_80[0:industry_cut[5]])]
target_test_d_10=np.array(data_test_i.iloc[:,1:-1])
target_test_l_10=np.array(data_test_i.iloc[:,-1])
target_test_d_10=torch.tensor(target_test_d_10).to(dtype=torch.float32)
target_test_l_10=torch.tensor(target_test_l_10).to(dtype=torch.float32)
dataset_10=TensorDataset(target_test_d_10,target_test_l_10)
target_test_loader_10=DataLoader(dataset=dataset_10,batch_size=1)
for i, dataset in enumerate(target_test_loader_10):
    data,label=dataset
prediction_10=prediction(target_test_loader_10)
prediction(target_test_loader_10)


loss_lambda=[prediction_val[4],prediction_80[4],prediction_60[4],prediction_40[4],prediction_30[4],prediction_20[4],
      prediction_10[4]]
print(loss_lambda)

recall_f1=[prediction_val[1],prediction_val[3]
           ,prediction_80[1],prediction_80[3]
           ,prediction_60[1],prediction_60[3]
           ,prediction_40[1],prediction_40[3]
           ,prediction_30[1],prediction_30[3]
           ,prediction_20[1],prediction_20[3]
           ,prediction_10[1],prediction_10[3]]
print(recall_f1)