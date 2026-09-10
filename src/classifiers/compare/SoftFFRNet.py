# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 16:14:27 2025

@author: YeZhang
"""

# encoding=utf-8
import torch
import torch.nn as nn
import torch.fft
import torch.utils.data as Data
import numpy as np
import time
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from utils.utils import *
from typing import Dict, List, Any

# ==========================================
# Soft-FFRNet Model Components
# ==========================================

class FourierFeatureRefiner(nn.Module):
    """
    Fourier Feature Refiner (FFR)
    Paper Section II.B: Extracts and refines features in the frequency domain.
    """
    def __init__(self, channels, length):
        super(FourierFeatureRefiner, self).__init__()
        # Learnable parameters alpha (amplitude) and beta (phase)
        # Shape: [1, Channels, Length] for element-wise multiplication
        self.alpha = nn.Parameter(torch.ones(1, channels, length))
        self.beta = nn.Parameter(torch.ones(1, channels, length))

    def forward(self, x):
        # x shape: [Batch, Channels, Length]
        
        # 1. Discrete Fourier Transform (DFT)
        fft_x = torch.fft.fft(x, dim=-1)
        
        # 2. Decompose into Amplitude and Phase
        amplitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # 3. Refine Amplitude and Phase (Eq. 7 & 8)
        refined_amplitude = amplitude * self.alpha
        refined_phase = phase * self.beta
        
        # 4. Reconstruction (Polar to Cartesian)
        # Eq. 9: REC = A' * e^(j * phi')
        real = refined_amplitude * torch.cos(refined_phase)
        imag = refined_amplitude * torch.sin(refined_phase)
        refined_complex = torch.complex(real, imag)
        
        # 5. Inverse Discrete Fourier Transform (IDFT)
        # Take the real part to return to the time domain
        refined_x = torch.fft.ifft(refined_complex, dim=-1).real
        
        return refined_x

class SoftThresholding(nn.Module):
    """
    Soft Thresholding Mechanism
    Paper Section II.C: Adaptive noise reduction.
    """
    def __init__(self, channels, reduction=8):
        super(SoftThresholding, self).__init__()
        # Two-layer FCN to determine the scaling factor rho (Eq. 16)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid() 
        )

    def forward(self, x):
        # x shape: [Batch, Channels, Length]
        batch, channels, length = x.shape
        
        # 1. Calculate Global Average of Absolute Values (theta)
        # Eq. 15: theta_c = sum(|x|) over length (Simplification of average logic)
        abs_x = torch.abs(x)
        theta = torch.sum(abs_x, dim=2) # Shape: [Batch, Channels]
        
        # 2. Calculate Scaling Factor (rho)
        rho = self.fc(theta) # Shape: [Batch, Channels]
        
        # 3. Calculate Threshold (tau)
        # Eq. 17: tau = rho * theta
        tau = rho * theta 
        tau = tau.unsqueeze(2) # Expand to [Batch, Channels, 1]
        
        # 4. Apply Soft Thresholding (Eq. 13/14)
        output = torch.sign(x) * torch.clamp(abs_x - tau, min=0)
        
        return output

class SoftResBlock(nn.Module):
    """
    Soft Thresholding-Based Residual Block
    Paper Section II.C and Fig. 2
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(SoftResBlock, self).__init__()
        
        # Convolution layer (Kernel=3, Stride=2/1, Padding=1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
        # Soft Thresholding is applied AFTER addition, and ReLU is removed inside the block as per paper
        self.soft_thresholding = SoftThresholding(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        
        # Residual Connection
        out = out + self.shortcut(x)
        
        # Soft Thresholding Activation
        out = self.soft_thresholding(out)
        
        return out

class SoftFFRNet(nn.Module):
    """
    Soft-FFRNet: Fourier Feature Refiner Network with Soft Thresholding
    Paper Fig. 1
    """
    def __init__(self, num_classes, input_length=1024, in_channels=1):
        super(SoftFFRNet, self).__init__()
        
        # 1. Initial Convolution
        # Kernel=3, Stride=2, Padding=1
        self.first_layer = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        
        # Calculate feature length after first stride=2 conv
        # Used for initializing FFR parameters
        self.ffr_length = input_length // 2
        
        # 2. Fourier Feature Refiner
        self.ffr = FourierFeatureRefiner(channels=16, length=self.ffr_length)
        
        # 3. Stacked Soft Thresholding Residual Blocks
        # Channels: 16 -> 32 -> 64 -> 128 -> 256 -> 512
        self.layer1 = SoftResBlock(16, 32, stride=2)
        self.layer2 = SoftResBlock(32, 64, stride=2)
        self.layer3 = SoftResBlock(64, 128, stride=2)
        self.layer4 = SoftResBlock(128, 256, stride=2)
        self.layer5 = SoftResBlock(256, 512, stride=2)
        
        # 4. Classification Layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Ensure input is 3D [Batch, Channel, Length]
        if x.dim() == 4:
            x = x.squeeze(1)
            
        # First Conv
        x = self.first_layer(x)
        
        # Fourier Feature Refiner
        x = self.ffr(x)
        
        # Residual Blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        # Classifier
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        
        # Return out, out to match the signature of other classifiers in the project (e.g. Resnet18.py)
        return out, out

# ==========================================
# Training and Evaluation Ops (Consistent with Project)
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