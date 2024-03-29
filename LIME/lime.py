#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 09:20:42 2023

@author: jolie
"""

import lime
import torch
import numpy as np
from torch import nn
import pandas as pd
import numpy as np
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer

######LIME
x_train=data_train.iloc[:,1:-1].values
y_train=data_train.iloc[:,-1].values
x_test=data_test.iloc[:,1:-1].values
y_test=data_test.iloc[:,-1].values
def batch_predict(data,model=model):
    X_tensor=torch.from_numpy(data).float()
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    X_tensor=X_tensor.to(device)
    
    logits,_=model(X_tensor,X_tensor)
    probs=F.softmax(logits,dim=1)
    
    return probs.detach().cpu().numpy()


class_names=['non-default','default']
feature_names=data_train.columns[1:-1]
explainer = LimeTabularExplainer(x_test,feature_names=feature_names.values, class_names=class_names,discretize_continuous=True)
exp = explainer.explain_instance(x_test[3808], batch_predict,num_features=21)
exp.show_in_notebook()
fig = exp.as_pyplot_figure() 