#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:19:08 2023

@author: jolie
"""

###Define Model

from __future__ import division
import warnings
import torch
from torch import nn
import numpy as np
warnings.filterwarnings("ignore")
import math


def init_weights(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
		nn.init.kaiming_uniform_(m.weight)
		nn.init.zeros_(m.bias)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight, 1.0, 0.02)
		nn.init.zeros_(m.bias)
	elif classname.find('Linear') != -1:
		nn.init.xavier_normal_(m.weight)
		nn.init.zeros_(m.bias)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
	return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
	def fun1(grad):
		return -coeff * grad.clone()

	return fun1

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class baseNetwork(nn.Module):
    def __init__(self, num_classes=2):
        super(baseNetwork, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(21,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
						nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
						#nn.Linear(128,128),
            #nn.ReLU(inplace=True),
            )
        
        self.bottleneck = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            )

        self.fc8 = nn.Sequential(
            nn.Linear(64, 2)
            )

    def forward(self, source): # computes activations for BOTH domains
        features = self.classifier(source)
        features = self.bottleneck(features)
        outputs = self.fc8(features)
        
        return features, outputs




class AdversarialNetwork(nn.Module):
	"""
    AdversarialNetwork obtained from official CDAN repository:
    https://github.com/thuml/CDAN/blob/master/pytorch/network.py
    """
	def __init__(self, in_feature, hidden_size):
		super(AdversarialNetwork, self).__init__()

		self.ad_layer1 = nn.Linear(in_feature, hidden_size)
		self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
		self.ad_layer3 = nn.Linear(hidden_size, 1)
		self.relu1 = nn.ReLU()
		self.relu2 = nn.ReLU()
		self.dropout1 = nn.Dropout(0.5)
		self.dropout2 = nn.Dropout(0.5)
		self.sigmoid = nn.Sigmoid()
		self.apply(init_weights)
		self.iter_num = 0
		self.alpha = 10
		self.low = 0.0
		self.high = 1.0
		self.max_iter = 10000.0

	def forward(self, x):
		#print("inside ad net forward",self.training)
		if self.training:
			self.iter_num += 1
		coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
		x = x * 1.0
		x.register_hook(grl_hook(coeff))
		x = self.ad_layer1(x)
		x = self.relu1(x)
		x = self.dropout1(x)
		x = self.ad_layer2(x)
		x = self.relu2(x)
		x = self.dropout2(x)
		y = self.ad_layer3(x)
		y = self.sigmoid(y)
		return y


	def output_num(self):
		return 1

	def get_parameters(self):
		return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]