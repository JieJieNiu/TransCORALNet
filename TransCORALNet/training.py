#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:07:31 2023

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
from save_log import save_log, save_model, load_model, get_mean_std_dataset


###Learning rate decay
def step_decay(epoch, learning_rate):
    """
    learning rate step decay
    :param epoch: current training epoch
    :param learning_rate: initial learning rate
    :return: learning rate after step decay
    """
    initial_lrate = learning_rate
    drop = 0.8
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate

###initialization
model=TransCORALNet()

def initialize(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data) 
model.apply(initialize)

class_weights=torch.tensor([0.25,0.75])
clf_criterion = nn.CrossEntropyLoss(weight=class_weights)#weight=class_weights
BATCH_SIZE = 256
TRAIN_EPOCHS = 250
learning_rate = 0.1
L2_DECAY = 0.0005
MOMENTUM = 0.9
training_statistic = []
testing_t_statistic = []

###Train
def train_ddcnet(epoch, model, LEARNING_RATE, source_loader, target_train_loader):
    """
    train source and target domain on ddcnet
    :param epoch: current training epoch
    :param model: defined ddcnet
    :param learning_rate: initial learning rate
    :param source_loader: source loader
    :param target_loader: target train loader
    :return:
    """
    log_interval = 500
    LEARNING_RATE = step_decay(epoch, learning_rate)

    optimizer = torch.optim.SGD([
        {"params": model.transformer.parameters()},
        {"params": model.classifier.parameters(),  "lr":LEARNING_RATE},
        {"params": model.final_classifier.parameters(), "lr":LEARNING_RATE},
    ], lr=LEARNING_RATE/10, momentum=MOMENTUM, weight_decay=L2_DECAY)
    
    print('learning rate',LEARNING_RATE)
    
    # enter training mode
    model.train()

    iter_source = iter(source_loader)
    iter_target_train = iter(target_train_loader)
    num_iter = max(len(source_loader),len(target_train_loader))
    correct_source = 0
    total_loss = 0
    clf_criterion = nn.CrossEntropyLoss(weight=class_weights)###weight=class_weights
    FN=0
    TN=0
    FP=0
    TP=0
    results=[]

    for i in range(0, num_iter):
        source_data,source_label = next(iter_source)
        target_train_data, target_train_label = next(iter_target_train)
     
        if i % len(target_train_loader) == 0:
            iter_target_train = iter(target_train_loader)
        if i % len(source_loader)==0:
            iter_source=iter(source_loader)

        source_data= Variable(source_data)
        source_label = Variable(source_label)
        target_train_data = Variable(target_train_data)
        target_train_label=Variable(target_train_label)
        
        lambda_factor = (epoch+1)/TRAIN_EPOCHS

        optimizer.zero_grad()

        source_preds,coral_loss= model(source_data, target_train_data)
        preds_source = source_preds.data.max(1, keepdim=True)[1]
        zes=Variable(torch.zeros(BATCH_SIZE).type(torch.LongTensor))
        ons=Variable(torch.ones(BATCH_SIZE).type(torch.LongTensor))
        train_correct01 = ((preds_source.squeeze(1)==zes)&(source_label==ons)).sum()
        train_correct10 = ((preds_source.squeeze(1)==ons)&(source_label==zes)).sum()
        train_correct11 = ((preds_source.squeeze(1)==ons)&(source_label==ons)).sum()
        train_correct00 = ((preds_source.squeeze(1)==zes)&(source_label==zes)).sum()
        FN += train_correct01
        FP += train_correct10
        TP += train_correct11
        TN += train_correct00
        correct_source += preds_source.eq(source_label.data.view_as(preds_source)).sum()
        
        ##Loss
        clf_loss_source = clf_criterion(source_preds,source_label.to(dtype=torch.long))
        loss =clf_loss_source + lambda_factor* coral_loss

        # compute gradients of network (backprop in pytorch)
        loss.backward()
        # update weights of network
        optimizer.step()
        
        results.append({
            'epoch': epoch,
            'step': i + 1,
            'total_steps': num_iter,
            'coral_loss': coral_loss.item(), # coral_loss.data[0],
            'classification_loss': clf_loss_source.item(),  # classification_loss.data[0],
            'total_loss': loss.item() # total_loss.data[0]
        })
        

        if i % log_interval == 0:
            print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tcoral_Loss: {:.6f}\lambda:{:.6f}'.format(
                epoch, i * len(source_data), len(source_loader) * BATCH_SIZE,
                100. * i / len(source_loader), loss, clf_loss_source, coral_loss,lambda_factor))

    
    total_loss /= len(source_loader)
    acc_source = float(correct_source) * 100. / (len(source_loader) * BATCH_SIZE)
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    F1=2*(recall*precision)/(recall+precision)                              

    print('Source_set: Average classification loss: {:.4f}, Accuracy source:{}/{} ({:.2f}%),Recall:{}, F1:{},Precision:{}'.format(
       total_loss, correct_source, len(source_loader.dataset),acc_source,recall,F1,precision))

    
    return results


def test_ddcnet(model, target_test_loader):
    """
    test target data on fine-tuned alexnet
    :param model: trained alexnet on source data set
    :param target_loader: target dataloader
    :return: correct num
    """
    # enter evaluation mod

    model.eval()
    test_loss = 0
    results_test=[]
    correct= 0
    FN_test= 0
    TN_test= 0
    FP_test= 0
    TP_test= 0

    with torch.no_grad():
        for test_data, test_target in target_test_loader:
            test_data, test_target = Variable(test_data), Variable(test_target)
            target_test_preds, _ = model(test_data, test_data)
        
            test_loss += clf_criterion(target_test_preds,test_target.to(dtype=torch.long)).item()# sum up batch loss
        
            test_pred = target_test_preds.data.max(1)[1] # get the index of the max log-probability
            correct += test_pred.eq(test_target.data.view_as(test_pred)).cpu().sum()
        
            zes=Variable(torch.zeros(BATCH_SIZE).type(torch.LongTensor))
            ons=Variable(torch.ones(BATCH_SIZE).type(torch.LongTensor))
            test_correct01 = ((test_pred==zes)&(test_target==ons)).sum()
            test_correct10 = ((test_pred==ons)&(test_target==zes)).sum()
            test_correct11 = ((test_pred==ons)&(test_target==ons)).sum()
            test_correct00 = ((test_pred==zes)&(test_target==zes)).sum()
            FN_test += test_correct01.item()
            FP_test += test_correct10.item()
            TP_test += test_correct11.item()
            TN_test += test_correct00.item()

        test_loss =test_loss/len(target_test_loader.dataset)
        acc_test=100.*correct/len(target_test_loader.dataset)
        recall_test=TP_test/(TP_test+FN_test)
        percision_test=TP_test/(TP_test+FP_test+1)
        F1_test=2*(recall_test*percision_test)/(recall_test+percision_test+0.01)
    
        results_test.append({
            "epoch": epoch+1,
            "average_loss": test_loss,
            "correct_class": correct.item(),
            "total_elems": len(target_test_loader.dataset),
            "accuracy_test": float(acc_test)
        })
    
    
        return correct,results_test

if __name__ == '__main__':


    
    source_loader=source_loader
    target_train_loader = target_train_loader
    target_test_loader = target_test_loader
    print('Load data complete')

    model = TransCORALNet(num_classes=2)
    print('Construct model complete')

    # # load pretrained alexnet model
    # ddcnet = model.load_pretrained_alexnet(ddcnet)
    # print('Load pretrained alexnet parameters complete\n')

    for epoch in range(1, TRAIN_EPOCHS + 1):
        print(f'Train Epoch {epoch}:')
        results=train_ddcnet(epoch, model, learning_rate, source_loader, target_train_loader)
        training_statistic.append(results)
        correct_test,results_test = test_ddcnet(model, target_test_loader)
        testing_t_statistic.append(results_test)
    
    save_log(training_statistic, 'adaptation_training_statistic.pkl')
    save_log(testing_t_statistic, 'adaptation_testing_t_statistic.pkl')
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    model_save_path = os.path.join('model.pt')
    torch.save(model.state_dict(), model_save_path)