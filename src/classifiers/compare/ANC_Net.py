# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 10:44:12 2025

@author: YeZhang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import math
import time
from utils.utils import *
import os

# ANC-Net: A novel multi-scale active noise cancellation network
# Reference: Expert Systems With Applications 265 (2025) 125937

class HaarWaveletPooling(nn.Module):
    """
    DWT Pooling Layer implementation using Haar Wavelet.
    Extracts the approximate (Low-frequency) component and downsamples by 2.
    Formula: L[i] = (x[2i] + x[2i+1]) / sqrt(2)
    """
    def __init__(self, in_channels):
        super(HaarWaveletPooling, self).__init__()
        self.in_channels = in_channels
        # Haar low-pass filter weights: [1/sqrt(2), 1/sqrt(2)]
        weight = torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)], dtype=torch.float32)
        # Reshape to [Out_channels, In_channels/Groups, Kernel_size]
        # Here we use groups=in_channels, so shape is [In_channels, 1, 2]
        self.register_buffer('weight', weight.view(1, 1, 2).repeat(in_channels, 1, 1))
        
    def forward(self, x):
        # x shape: [Batch, Channel, Length]
        # Padding to handle odd lengths
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1))
        # Convolution with stride 2 implements the DWT decomposition + downsampling
        out = F.conv1d(x, self.weight, stride=2, groups=self.in_channels)
        return out

class SelfAttention(nn.Module):
    """
    Channel-wise Self-Attention (SE-Block style) adapted for 1D signals.
    """
    def __init__(self, channels, reduction=16):
        super(SelfAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # Reduction ratio logic ensures strictly positive hidden dimension
        mid_channels = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """
    Small/Middle/Large scale Residual Block with Self-Attention.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.selu = nn.SELU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        # self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.attention = SelfAttention(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.selu(out)
        
        out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.selu(out)
        
        # Apply attention before residual addition
        out = self.attention(out)
        
        out = out + identity
        return out

class MultiScaleBranch(nn.Module):
    """
    A single branch processing features at a specific scale.
    Structure: ResBlock -> BN -> SELU -> DWT Pooling
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MultiScaleBranch, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.selu = nn.SELU(inplace=True)
        self.dwt_pool = HaarWaveletPooling(out_channels)

    def forward(self, x):
        x = self.res_block(x)
        x = self.bn(x)
        x = self.selu(x)
        x = self.dwt_pool(x)
        return x

class MultiScaleBlock(nn.Module):
    """
    Extracts and fuses features from 3 scales (kernels: 3, 7, 17).
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        # 3 parallel branches
        self.branch_s = MultiScaleBranch(in_channels, out_channels, kernel_size=3)
        self.branch_m = MultiScaleBranch(in_channels, out_channels, kernel_size=7)
        self.branch_l = MultiScaleBranch(in_channels, out_channels, kernel_size=17)
        
        # Post fusion processing: BN -> SELU -> DWT Pooling
        self.post_process = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.SELU(inplace=True),
            HaarWaveletPooling(out_channels)
        )

    def forward(self, x):
        # Extract features
        s = self.branch_s(x)
        m = self.branch_m(x)
        l = self.branch_l(x)
        
        # Element-wise fusion (Summation)
        fused = s + m + l
        
        # Post processing
        out = self.post_process(fused)
        return out

class ANC_Net(nn.Module):
    def __init__(self, num_classes, kernel_size=None):
        # Note: kernel_size argument is kept for compatibility with the project's instantiation logic,
        # but ANC-Net uses fixed multi-scale kernels (3, 7, 17) internally.
        super(ANC_Net, self).__init__()
        
        self.num_classes = num_classes
        
        # 1. Initial Convolution
        # Assuming input has 1 channel (raw vibration signal)
        self.initial_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.SELU(inplace=True)
        )
        
        # 2. Multi-scale Feature Extractor
        # Block 1: 32 -> 32 channels, performs 4x downsampling (2x inside branch, 2x after fusion)
        self.ms_block1 = MultiScaleBlock(32, 32)
        # Block 2: 32 -> 32 channels, performs another 4x downsampling
        self.ms_block2 = MultiScaleBlock(32, 32)
        
        self.attention_fuse = SelfAttention(32)
        
        # 3. Fault Predictor
        self.adp_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 128),
            nn.Dropout(0.5),
            nn.SELU(inplace=True),
            nn.Linear(128, 128),
            nn.Dropout(0.5),
            nn.SELU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Ensure input is [Batch, 1, Length] for 1D Conv
        if x.dim() == 4:
            # If input is [Batch, 1, 1, Length] or [Batch, 1, H, W] from typical image loaders
            b, c, h, w = x.size()
            x = x.view(b, c, -1)
            
        feat = self.initial_conv(x)
        feat = self.ms_block1(feat)
        feat = self.ms_block2(feat)
        
        feat = self.attention_fuse(feat)
        
        feat = self.adp_pool(feat)
        
        features = self.flatten(feat)
        
        logits = self.classifier(features)
        
        # Return logits and features (for consistency with WDCNN training loop)
        return logits, features

# ==========================================================================================
# Training and Testing Operations (Adapted from WDCNN.py)
# ==========================================================================================

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
                                                           patience=5,
                                                           min_lr=LR/100)
    # loss_function = LabelSmoothingCrossEntropy()
    loss_function = nn.CrossEntropyLoss()
    # save init model
    output_directory_init = output_directory_models + 'init_model.pkl'
    torch.save(network.state_dict(), output_directory_init)  # only save the initialized parameters

    training_duration_logs = []
    start_time = time.time()
    for epoch in range(EPOCH):

        for step, (x, y) in enumerate(train_loader):
            # h_state = None      # for initial hidden state

            batch_x = x.cuda()
            batch_y = y.cuda()
            output_bc = network(batch_x)[0]

            # cal the sum of pre loss per batch
            loss = loss_function(output_bc, batch_y)
            optimizer.zero_grad()
            loss.backward()
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