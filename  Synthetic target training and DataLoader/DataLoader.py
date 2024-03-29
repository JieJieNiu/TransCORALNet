#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:58:39 2023

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


###dataloader
####Load data to tensor
special_80=pd.read_csv('special_80.csv')
special_80=list(special_80['行业80'])
data_test=pd.read_csv('data_newuser_80.csv')
data_train=pd.read_csv('data_train.csv')
fake_data=pd.read_csv('fake_data.csv')

data_test=data_test.iloc[:,1:]
data_train=data_train.iloc[:,1:]
fake_data=fake_data.iloc[:,1:]

x = data_train.iloc[0:82176,1:-1]
y = data_train.iloc[0:82176, -1]
source_d, validation_d, source_l, validation_l = train_test_split(x, y, test_size=0.2,random_state=29)
target_test_d=np.array(data_test.iloc[0:5632,1:-1])
target_test_l=np.array(data_test.iloc[0:5632,-1])
target_train_d=np.array(fake_data.iloc[0:65792,1:-1])
target_train_l=np.array(fake_data.iloc[0:65792,-1])


source_d=torch.tensor(np.array(source_d)).to(dtype=torch.float32)
source_l=torch.tensor(np.array(source_l)).to(dtype=torch.float32)
target_train_d=torch.tensor(target_train_d).to(dtype=torch.float32)
target_train_l=torch.tensor(target_train_l).to(dtype=torch.float32)
target_test_d=torch.tensor(target_test_d).to(dtype=torch.float32)
target_test_l=torch.tensor(target_test_l).to(dtype=torch.float32)
source_l.size()

dataset=TensorDataset(source_d,source_l)
source_loader=DataLoader(dataset=dataset,batch_size=256,shuffle=True)

for i, dataset in enumerate(source_loader):
    data,label=dataset


dataset=TensorDataset(target_train_d,target_train_l)
target_train_loader=DataLoader(dataset=dataset,batch_size=256,shuffle=True)
for i, dataset in enumerate(target_train_loader):
    data,label=dataset

dataset=TensorDataset(target_test_d,target_test_l)
target_test_loader=DataLoader(dataset=dataset,batch_size=256,shuffle=True)
for i, dataset in enumerate(target_test_loader):
    data,label=dataset