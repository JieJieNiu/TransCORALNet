#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:00:49 2023

@author: jolie
"""

import torch
import numpy as np
import scipy as sp
from torch import nn
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import TensorDataset, DataLoader
import math
from torch.nn import init
import torch.nn.functional as F


checkpoint = torch.load('model_TransCORALNet.pt')
model = TransCORALNet()
model.load_state_dict(checkpoint)
weight=checkpoint['transformer.0.self_attn.in_proj_weight']
bias=checkpoint['transformer.0.self_attn.in_proj_bias']

wq=weight[0:21] #weight of quary
wk=weight[21:42] #weight of key
wv=weight[42:63] #weight of value
bq=bias[0:21]  #bias of quary
bk=bias[21:42] #bias of key
bv=bias[42:63] #bias of value


##attention score
inputx=torch.tensor(source_d).to(dtype=torch.float32).unsqueeze(1)
q=inputx*wq+bq
k=inputx*wk+bk
att=torch.matmul(q,k.transpose(1,2))/np.sqrt(21)
att = torch.softmax(att, -1)
score = torch.zeros(21,21)
for i in range(len(att)):
    score+=att[i]



####Calculate Label0 and Label1的attention score difference
neg=np.array(data_test[data_test["Default_label"]==0].iloc[:,1:-1])
pos=data_test[data_test["Default_label"]==1].iloc[:,1:-1]
neg=np.array(neg,dtype=np.float)
pos=np.array(pos)
neg=torch.tensor(neg).to(dtype=torch.float32)
pos=torch.tensor(pos).to(dtype=torch.float32)

inputx=torch.tensor(pos).to(dtype=torch.float32).unsqueeze(1)
q=inputx*wq+bq
k=inputx*wk+bk
att=torch.matmul(q,k.transpose(1,2))/np.sqrt(21)
att = torch.softmax(att, -1)
score_pos = torch.zeros(21,21)
for i in range(len(att)):
    score_pos+=att[i]
    
inputx=torch.tensor(neg).to(dtype=torch.float32).unsqueeze(1)
q=inputx*wq+bq
k=inputx*wk+bk
att=torch.matmul(q,k.transpose(1,2))/np.sqrt(21)
att = torch.softmax(att, -1)
score_neg = torch.zeros(21,21)
for i in range(len(att)):
    score_neg+=att[i]
    
    
####Plot
import seaborn as sns
def plotattentionscore(data):
    inputx=torch.tensor(data).to(dtype=torch.float32).unsqueeze(1)
    q=inputx*wq+bq
    k=inputx*wk+bk
    att=torch.matmul(q,k.transpose(1,2))/np.sqrt(21)
    att = torch.softmax(att, -1)
    score = torch.zeros(21,21)
    for i in range(len(att)):
        score+=att[i]
    score=np.array(score/len(data))
    sns.set(font_scale=0.8)
    sns.heatmap(score, linewidths=.5,robust=True,cmap=sns.cubehelix_palette(as_cmap=True),xticklabels=feature_names,yticklabels=feature_names)
    return score
    
neg=data_train[data_train["Default_label"]==0].iloc[:,1:-1]
pos=data_train[data_train["Default_label"]==1].iloc[:,1:-1]
neg=np.array(neg,dtype=np.float)
pos=np.array(pos)