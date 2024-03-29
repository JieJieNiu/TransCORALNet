#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:56:09 2023

@author: jolie
"""

###Generate fake data
from sdv.demo import load_tabular_demo
from sdv.tabular import CTGAN
from table_evaluator import load_data, TableEvaluator
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold,datasets
import pandas as pd
from sklearn import preprocessing
from  sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from numpy import where

####CTGAN
data_newuser=pd.read_csv('data_newuser_80.csv')

#the row number of generated data
new_data_rows=65792
#generate new data
ctgan = CTGAN()
ctgan.fit(data_newuser.iloc[:,1:-1])
fake_data= ctgan.sample(num_rows=new_data_rows)

#Evaluator
table_evaluator = TableEvaluator(data_newuser.iloc[:,1:-1], fake_data)
table_evaluator.visual_evaluation()