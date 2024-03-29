#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:02:50 2023

@author: jolie
"""

from skimage import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms, datasets
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
def save_log(obj, path):
	with open(path, 'wb') as f:
		pickle.dump(obj, f)
		print('[INFO] Object saved to {}'.format(path))

def save_model(model, path):
	torch.save(model.state_dict(), path)
	print("checkpoint saved in {}".format(path))

def load_model(model, path):
	"""
	Loads trained network params in case AlexNet params are not loaded.
	"""
	model.load_state_dict(torch.load(path))
	print("pre-trained model loaded from {}".format(path))

def get_mean_std_dataset(root_dir):
	"""
	Function to compute mean and std of image dataset.
	Move batch_size param according to memory resources.
	retrieved from: https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
	"""

	# data_domain = "amazon"
	# path_dataset = "datasets/office/%s/images" % data_domain

	transform = transforms.Compose([
	transforms.Resize((224, 224)), # original image size 300x300 pixels
	transforms.ToTensor()])

	dataset = datasets.ImageFolder(root=root_dir,
	                               transform=transform)

	# set large batch size to get good approximate of mean, std of full dataset
	# batch_size: 4096, 2048
	data_loader = DataLoader(dataset, batch_size=2048,
	                        shuffle=False, num_workers=0)

	mean = []
	std = []

	for i, data in enumerate(data_loader, 0):
	    # shape is (batch_size, channels, height, width)
	    npy_image = data[0].numpy()

	    # compute mean, std per batch shape (3,) three channels
	    batch_mean = np.mean(npy_image, axis=(0,2,3))
	    batch_std = np.std(npy_image, axis=(0,2,3))

	    mean.append(batch_mean)
	    std.append(batch_std)

	# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
	mean = np.array(mean).mean(axis=0) # average over batch averages
	std = np.arry(std).mean(axis=0) # average over batch stds

	values = {
	    "mean": mean,
	    "std": std
	}

	return values