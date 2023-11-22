#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:05:04 2023

@author: jolie
"""

###Model
from torch import nn

class TransCORALNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AADeepCORAL, self).__init__()
        self.transformer = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=21, nhead=7,dim_feedforward=32,batch_first=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(21, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )

        self.final_classifier = nn.Sequential(
            nn.Linear(64, 2)
        )

    def forward(self, source, target):
        source = self.transformer(source)
        source = self.classifier(source)

        coral_loss = 0
        if self.training:
            target = self.transformer(target)
            target = self.classifier(target)
            coral_loss += CORAL(source, target)

        source = self.final_classifier(source)
        
        return source, coral_loss