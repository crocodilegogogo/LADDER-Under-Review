# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 17:45:10 2025

@author: YeZhang
"""

# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import pandas as pd
import time
from utils.utils import *
from typing import Dict, List, Any
import os

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

# ==========================================
# Model Architecture Components
# ==========================================

class CausalConv1d(nn.Module):
    """
    1D Causal Convolution implementation.
    Padding is applied only to the left to ensure causality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=0,  # We handle padding manually
            dilation=dilation, 
            bias=bias
        )

    def forward(self, x):
        # Apply padding to the left side only (Time dimension)
        # x shape: [Batch, Channel, Length]
        x = F.pad(x, (self.padding, 0)) 
        return self.conv(x)

class MSDC_Branch(nn.Module):
    """
    Single Branch of the Multi-scale Dilated Convolution (MSDC) Module.
    """
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(MSDC_Branch, self).__init__()
        
        # Conv1d for dimensionality expansion
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.01)
        
        # Two layers of dilated convolutions
        self.dconv1 = nn.Conv1d(out_channels, out_channels, kernel_size=5, 
                                padding=2*dilation_rate, dilation=dilation_rate)
        self.dconv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, 
                                padding=2*dilation_rate, dilation=dilation_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dconv1(x)
        x = self.dconv2(x)
        return x

class MSDC_Module(nn.Module):
    """
    Multi-scale Dilated Convolution Module.
    Consists of 3 parallel branches with dilations [1, 2, 4].
    """
    def __init__(self, in_channels=21):
        super(MSDC_Module, self).__init__()
        # Table 1: MSDC (each branch) out_channel=21
        self.branch1 = MSDC_Branch(in_channels, 21, dilation_rate=1)
        self.branch2 = MSDC_Branch(in_channels, 21, dilation_rate=2)
        self.branch3 = MSDC_Branch(in_channels, 21, dilation_rate=4)
        
        self.bn_final = nn.BatchNorm1d(21 * 3) # 63 channels
        self.act = nn.LeakyReLU(0.01)

    def forward(self, x):
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        
        # Depth Concatenation
        out = torch.cat([y1, y2, y3], dim=1)
        out = self.bn_final(out)
        out = self.act(out)
        return out

class DenseTCN_Layer(nn.Module):
    """
    Single Layer within a DenseTCN Block.
    """
    def __init__(self, in_channels, growth_rate, base_dilation):
        super(DenseTCN_Layer, self).__init__()
        
        # Bottleneck / Dimension Alignment
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.act = nn.LeakyReLU(0.01)
        self.conv1 = nn.Conv1d(in_channels, inter_channels, kernel_size=1)
        
        # Three layers of Dilated Causal Convolutions
        self.bn2 = nn.BatchNorm1d(inter_channels)
        
        d1 = base_dilation
        d2 = base_dilation * 2
        d3 = base_dilation * 3
        
        self.causal_conv1 = CausalConv1d(inter_channels, inter_channels, kernel_size=3, dilation=d1)
        self.causal_conv2 = CausalConv1d(inter_channels, inter_channels, kernel_size=3, dilation=d2)
        self.causal_conv3 = CausalConv1d(inter_channels, inter_channels, kernel_size=3, dilation=d3)
        
        # Final projection to growth_rate
        self.conv_out = nn.Conv1d(inter_channels, growth_rate, kernel_size=1)

    def forward(self, x):
        # Pre-activation design (DenseNet style)
        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.act(out)
        
        # Cascade of dilated causal convs
        out = self.causal_conv1(out)
        out = self.causal_conv2(out)
        out = self.causal_conv3(out)
        
        out = self.conv_out(out)
        return out

class DenseTCN_Block(nn.Module):
    """
    DenseTCN Block consisting of N layers with dense connections.
    """
    def __init__(self, in_channels, growth_rate=32, n_layers=5, base_dilation=1):
        super(DenseTCN_Block, self).__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        
        for i in range(n_layers):
            layer = DenseTCN_Layer(current_channels, growth_rate, base_dilation)
            self.layers.append(layer)
            current_channels += growth_rate # Dense connection adds channels
            
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate all previous features as input to next layer
            layer_in = torch.cat(features, dim=1) 
            new_feature = layer(layer_in)
            features.append(new_feature)
        
        # Output is the concatenation of all features (Input + all layer outputs)
        return torch.cat(features, dim=1)

class MSDC_DenseTCN(nn.Module):
    """
    Complete MSDC-DenseTCN Model.
    """
    def __init__(self, num_classes=10, input_channels=1):
        super(MSDC_DenseTCN, self).__init__()
        
        # 1. Convolution Block
        # Conv1d: out=21, k=7, s=2 -> MaxPool: k=3, s=2
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 21, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(21), 
            nn.LeakyReLU(0.01),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # 2. MSDC Module
        # Input: 21 channels. Output: 63 channels (21*3).
        self.msdc = MSDC_Module(in_channels=21)
        
        # 3. Parallel DenseTCN Blocks
        # Input to blocks is MSDC output (63 channels).
        growth_rate = 32
        n_layers = 5 
        
        # Branch 1 (Base dilation 1)
        self.dense_block1 = DenseTCN_Block(63, growth_rate, n_layers, base_dilation=1)
        # Branch 2 (Base dilation 2)
        self.dense_block2 = DenseTCN_Block(63, growth_rate, n_layers, base_dilation=2)
        # Branch 3 (Base dilation 4)
        self.dense_block3 = DenseTCN_Block(63, growth_rate, n_layers, base_dilation=4)
        
        # Calculating Output Channels:
        # Each block input: 63
        # Each block generates: 5 layers * 32 = 160 channels
        # Each block output: 63 + 160 = 223 channels
        # Total parallel output: 223 * 3 = 669 channels.
        
        # 4. Feature Recognition / Classification
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(669, num_classes)

    def forward(self, x):
        # x: [Batch, 1, Length]
        if x.dim() == 4 and x.shape[2] == 1: # Handle [B, 1, 1, L] if present
             x = x.squeeze(2)
        
        # Conv Block
        x = self.conv_block(x)
        
        # MSDC
        x = self.msdc(x)
        
        # Parallel DenseTCN
        b1 = self.dense_block1(x)
        b2 = self.dense_block2(x)
        b3 = self.dense_block3(x)
        
        # Concatenate parallel block outputs
        out = torch.cat([b1, b2, b3], dim=1)
        
        # Global Pooling & Linear
        out = self.gap(out).squeeze(-1) # [Batch, 669]
        out = self.fc(out)              # [Batch, Num_Classes]
        
        return out, out

# ==========================================
# Training and Prediction Functions
# ==========================================

def train_op(network, EPOCH, BATCH_SIZE, LR,
             train_x, train_y, val_x, val_y,
             output_directory_models, log_training_duration, test_split):
    # prepare training_data
    if train_x.shape[0] % BATCH_SIZE == 1:
        drop_last_flag = True
    else:
        drop_last_flag = False
    torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
    train_loader = Data.DataLoader(dataset=torch_dataset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   drop_last=drop_last_flag
                                   )

    # init lr&train&test loss&acc log
    lr_results = []
    loss_train_results = []
    accuracy_train_results = []
    loss_validation_results = []
    accuracy_validation_results = []

    # prepare optimizer&scheduler&loss_function
    optimizer = torch.optim.Adam(network.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                           patience=10,
                                                           min_lr=LR/100)
    loss_function = nn.CrossEntropyLoss()


    # save init model
    output_directory_init = output_directory_models + 'init_model.pkl'
    torch.save(network.state_dict(), output_directory_init)  # save only the init parameters

    training_duration_logs = []
    start_time = time.time()
    for epoch in range(EPOCH):
        epoch_tau = epoch+1
        tau = max(1 - (epoch_tau - 1) / 100, 0.25)
        for m in network.modules():
            if hasattr(m, '_update_tau'):
                m._update_tau(tau)
                # print(a)

        for step, (x, y) in enumerate(train_loader):
            # h_state = None      # for initial hidden state

            batch_x = x.cuda()
            batch_y = y.cuda()
            output_bc, regus = network(batch_x)
            # cal the sum of pre loss per batch
            loss_class = loss_function(output_bc, batch_y)
            ### arguement
            loss_total = loss_class
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

        # test per epoch
        network.eval()
        # loss_train:loss of training set; accuracy_train:pre acc of training set
        loss_train, accuracy_train = get_test_loss_acc(network, loss_function, train_x, train_y, test_split)
        loss_validation, accuracy_validation = get_test_loss_acc(network, loss_function, val_x, val_y, test_split)
        network.train()

        # update lr
        # scheduler.step(loss_validation)
        scheduler.step(loss_train)
        lr = optimizer.param_groups[0]['lr']

        ######################################dropout#####################################
        # loss_train, accuracy_train = get_loss_acc(network.eval(), loss_function, train_x, train_y, test_split)

        # loss_validation, accuracy_validation = get_loss_acc(network.eval(), loss_function, test_x, test_y, test_split)

        # network.train()
        ##################################################################################

        # log lr&train&validation loss&acc per epoch
        lr_results.append(lr)
        loss_train_results.append(loss_train)
        accuracy_train_results.append(accuracy_train)
        loss_validation_results.append(loss_validation)
        accuracy_validation_results.append(accuracy_validation)

        # print training process
        if (epoch + 1) % 10 == 0:
            print('Epoch:', (epoch + 1), '|lr:', lr,
                  '| train_loss:', loss_train,
                  # '| train_loss:', loss_train,
                  # '| regu_loss:', loss_regu,
                  '| train_acc:', accuracy_train,
            '| validation_loss:', loss_validation,
            '| validation_acc:', accuracy_validation)

        save_models(network, output_directory_models,
                    loss_train, loss_train_results,
                    accuracy_validation, accuracy_validation_results,
                    )

    # log training time
    per_training_duration = time.time() - start_time
    log_training_duration.append(per_training_duration)

    # save last_model
    output_directory_last = output_directory_models + 'last_model.pkl'
    torch.save(network.state_dict(), output_directory_last)  # save only the init parameters

    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results,
                          loss_validation_results, accuracy_validation_results, output_directory_models)

    plot_learning_history(EPOCH, history, output_directory_models)

    return (history, per_training_duration, log_training_duration)