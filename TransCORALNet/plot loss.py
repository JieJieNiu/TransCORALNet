#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:05:37 2023

@author: jolie
"""

import pickle
import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import argparse


S=open(r'adaptation_training_statistic.pkl','rb')
T=open(r'adaptation_testing_t_statistic.pkl','rb')
adapt_training_dict=pickle.load(S)
adapt_testing_target_dict=pickle.load(T)

adaptation = {"classification_loss_train":[],
              "coral_loss_train":[],
              "source_accuracy":[],
              "target_accuracy":[],
              "classification_loss_test":[]}

for epoch_idx in range(len(adapt_training_dict)): # epoch
    coral_loss_train = 0
    class_loss = 0
    class_loss_test = 0
    
    for step_idx in range(len(adapt_training_dict[epoch_idx])):
        coral_loss_train += adapt_training_dict[epoch_idx][step_idx]["coral_loss"]
        class_loss += adapt_training_dict[epoch_idx][step_idx]["classification_loss"]
        
        for step_idx_t in range(len(adapt_testing_target_dict[epoch_idx])):
            class_loss_test += adapt_testing_target_dict[epoch_idx][step_idx_t]["average_loss"]
            
        
        # store average losses in general adaptation dictionary
    adaptation["classification_loss_train"].append(class_loss/len(adapt_training_dict[epoch_idx]))
    adaptation["coral_loss_train"].append(coral_loss_train/len(adapt_training_dict[epoch_idx]))
    adaptation["classification_loss_test"].append(class_loss_test/len(adapt_testing_target_dict[epoch_idx]))

fig=plt.figure(figsize=(8, 6), dpi=100)
fig.show()

plt.xlabel("epochs", fontsize=15)
plt.ylabel("loss", fontsize=15)

plt.plot(adaptation["classification_loss_train"], label="classification_loss_train", marker='*', markersize=8)


plt.legend(loc="best")
plt.grid()
plt.show()


fig2=plt.figure(figsize=(8, 6), dpi=100)
fig.show()

plt.xlabel("epochs", fontsize=15)
plt.ylabel("loss", fontsize=15)

plt.plot(adaptation["classification_loss_test"], label="classification_loss_test", marker='o', markersize=3)

plt.legend(loc="best")
plt.grid()
plt.show()



fig3=plt.figure(figsize=(8, 6), dpi=100)
fig.show()

plt.xlabel("epochs", fontsize=15)
plt.ylabel("loss", fontsize=15)

plt.plot(adaptation["coral_loss_train"], label="coral_loss", marker='o', markersize=3)

plt.legend(loc="best")
plt.grid()
plt.show()