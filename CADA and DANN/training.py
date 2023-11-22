#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:24:38 2023

@author: jolie
"""

from __future__ import division
import pandas as pd
import warnings
import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from torch.autograd import Variable
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from torch.nn import init
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from model import baseNetwork,AdversarialNetwork
from loss import DANN, CADA
from save_log import save_log, save_model, load_model


####Train model
# set model hyperparameters (paper page 5)
learning_rate = 0.01
L2_DECAY = 5e-4
MOMENTUM = 0.9
EPOCHS=150
log_interval=500
training_s_statistic =[]
testing_s_statistic = []
testing_t_statistic = []
results_test = []
results_train = []
class_weights=torch.tensor([0.25,0.75])
clf_criterion=nn.CrossEntropyLoss(weight=class_weights)


#实例化
model = baseNetwork()
ad_net = AdversarialNetwork(64,128)

def initialize(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data) 
model.apply(initialize)
ad_net.apply(initialize)


####Train
def train(model, ad_net, train_loader, target_train_loader, epoch, start_epoch):
    """
    This method fits the network params one epoch at a time.
    Implementation based on:
    https://github.com/SSARCandy/DeepCORAL/blob/master/main.py
    """
    model.train()
    correct_source = 0
    FN=0
    TN=0
    FP=0
    TP=0
    # LEARNING_RATE = step_decay(epoch, learning_rate)
    # optimizer = torch.optim.SGD([
    #     {"params": model.classifier.parameters(),"lr":LEARNING_RATE},
    #     {"params": model.bottleneck.parameters(),"lr":LEARNING_RATE},
    #     {"params": model.fc8.parameters(), "lr":LEARNING_RATE},
    #     {"params":ad_net.parameters(), "lr_mult": 10, 'decay_mult': 2}
    # ], lr=LEARNING_RATE,momentum=MOMENTUM)#momentum=MOMENTUM,
    # LEARNING_RATE = step_decay(epoch, learning_rate)
    LEARNING_RATE=learning_rate
    
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0005, momentum=0.9)
    optimizer_ad = optim.SGD(ad_net.parameters(), lr=LEARNING_RATE, weight_decay=0.0005, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY,betas=(0.9, 0.999))
    # optimizer_ad = optim.Adam(ad_net.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY,betas=(0.9, 0.999))

    iter_source = iter(source_loader)
    iter_target_train = iter(target_train_loader)
    train_steps =  min(len(source_loader),len(target_train_loader))

    # start batch training
    for i in range(0,train_steps):
        # fetch data in batches
        source_data,source_label = next(iter_source)
        target_data, target_label = next(iter_target_train)
        # ##保证数据能够整除Batch
        if i % len(target_train_loader) == 0:
            iter_target_train = iter(target_train_loader)
        if i % len(source_loader)==0:
            iter_source=iter(source_loader)

        # create pytorch variables, the variables and functions build a dynamic graph of computation
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        # reset to zero optimizer gradients
        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        
        lambda_factor=(epoch+1)/EPOCHS

        # do a forward pass through network (recall DeepCORAL outputs source, target activation maps)
        src_features, src_ouputs = model(source_data)
        tgt_features, tgt_ouputs = model(target_data)
        feature = torch.cat((src_features, tgt_features), dim=0)
        output = torch.cat((src_ouputs, tgt_ouputs), dim=0)
        softmax_output = nn.Softmax(dim=1)(output)
        # entropy = Entropy(softmax_output)
        pred = src_ouputs.data.max(1, keepdim=True)[1]

        classification_loss = clf_criterion(output.narrow(0, 0, source_data.size(0)), source_label.to(dtype=torch.long))
        
        
        zes=Variable(torch.zeros(BATCH_SIZE).type(torch.LongTensor))
        ons=Variable(torch.ones(BATCH_SIZE).type(torch.LongTensor))
        train_correct01 = ((pred.squeeze(1)==zes)&(source_label==ons)).sum() 
        train_correct10 = ((pred.squeeze(1)==ons)&(source_label==zes)).sum() 
        train_correct11 = ((pred.squeeze(1)==ons)&(source_label==ons)).sum() 
        train_correct00 = ((pred.squeeze(1)==zes)&(source_label==zes)).sum() 
        FN += train_correct01
        FP += train_correct10
        TP += train_correct11
        TN += train_correct00
        correct_source += pred.eq(source_label.data.view_as(pred)).cpu().sum()

        #transfer_loss = CDAN([feature, softmax_output], ad_net,None,None,None)
        transfer_loss = DANN(feature, ad_net)


        # compute total loss
        total_loss = classification_loss + lambda_factor*transfer_loss
        
        # compute gradients of network
        total_loss.backward()

        # update weights of network
        optimizer.step()
        
        if epoch > 1:
            optimizer_ad.step()
            
        # print training info
        if i % log_interval == 0:
            print('Train Epoch: {:2d} [{:2d}/{:2d}]\t'
                  'Classification loss: {:.6f}, transfer loss: {:.6f}, Total_Loss: {:.6f}'.format(
                      epoch,
                      i + 1,
                      train_steps,
                      # lambda_factor,
                      classification_loss.item(), # classification_loss.data[0],
                      transfer_loss.item(),
                      total_loss.item() # total_loss.data[0]
                  ))

        # append results for each batch iteration as dictionaries
        results_train.append({
            'epoch': epoch,
            'step': i+ 1,
            'total_steps': train_steps,
            'transfer_loss': transfer_loss.item(), 
            'classification_loss': classification_loss.item(),  # classification_loss.data[0],
            'total_loss': total_loss.item() # total_loss.data[0]
        })


    total_loss /= len(source_loader)
    acc_source = float(correct_source) * 100. / (len(source_loader) * BATCH_SIZE)
    recall=TP/(TP+FN)
    precision=TP/(TP+FP)
    F1=2*(recall*precision)/(recall+precision)
    

    print('Source_set: Average classification loss: {:.4f}, Accuracy source:{:.2f}%,Recall:{}, F1:{},Precision:{}'.format(
       total_loss, acc_source,recall,F1,precision))
    
    return results_train

def test(model, target_test_loader, epoch):
    """
    Computes classification accuracy of (labeled) data using cross-entropy.
    Retreived from: https://github.com/SSARCandy/DeepCORAL/blob/master/main.py
    """
    # eval() it indicates the model that nothing new is
    # to be learnt and the model is used for testing
    FN_test= 0
    TN_test= 0
    FP_test= 0
    TP_test= 0
    test_loss = 0
    correct_class=0
    model.eval()

    test_loss = 0
    correct_class = 0
    
    with torch.no_grad():

    # go over dataloader batches, labels
        for data, label in target_test_loader:

        # note on volatile: https://stackoverflow.com/questions/49837638/what-is-volatile-variable-in-pytorch
            data, label = Variable(data, volatile=True), Variable(label)
            feature, output = model(data) # just use one ouput of DeepCORAL

        # sum batch loss when computing classification
            test_loss += clf_criterion(output, label.to(dtype=torch.long)).item()
            # test_loss += nn.CrossEntropyLoss(weight=class_weights)(output, label.to(dtype=torch.long))

        # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct_class += pred.eq(label.data.view_as(pred)).sum().item()

    # compute test loss as correclty classified labels divided by total data size
        test_loss /= len(target_test_loader.dataset)
        acc_test=100.*correct_class/len(target_test_loader.dataset)
        
        # return dictionary containing info of each epoch
        results_test.append({
            "epoch": epoch,
            "average_loss": test_loss,
            "correct_class": correct_class,
            "total_elems": len(target_test_loader.dataset),
            "accuracy %": acc_test
        })
        print("Test_Accuracy:{}".format(acc_test))
    
        # print("Test_Accuracy:{}/tRecall:{}/tPrecision:{}/tF1:{}/tloss{}".format(acc_test,recall_test,percision_test,F1_test,test_loss))
        
    return results_test


if __name__ == '__main__':

    source_loader=source_loader
    target_train_loader = target_train_loader
    target_test_loader = target_test_loader
    
    # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY, momentum=MOMENTUM)
    # optimizer_ad = optim.SGD(ad_net.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY, momentum=MOMENTUM)


    # # load pretrained alexnet model
    # ddcnet = model.load_pretrained_alexnet(ddcnet)
    # print('Load pretrained alexnet parameters complete\n')

    for epoch in range(1, EPOCHS+1):

        # run batch trainig at each epoch (returns dictionary with epoch result)
        results_train = train(model, ad_net, source_loader, target_train_loader, epoch, 0)
        training_s_statistic.append(results_train)
        
        results_test = test(model, target_test_loader, epoch)
        testing_t_statistic.append(results_test)
    
    save_log(training_s_statistic, 'adaptation_training_statistic.pkl')
    save_log(testing_t_statistic, 'adaptation_testing_t_statistic.pkl')
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    model_save_path = os.path.join('model.pt')
    torch.save(model.state_dict(), model_save_path)
    
    