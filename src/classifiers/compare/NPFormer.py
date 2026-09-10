# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 23:22:31 2025

@author: YeZhang
"""

# encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import time
from utils.utils import *
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
# NPFormer Components
# ==========================================

class LargeConvBlock(nn.Module):
    """
    Section 3.2: Large kernel for feature extraction
    Structure: Conv(k=31, s=2) -> BN -> GELU
    """
    def __init__(self, in_channels, out_channels, kernel_size=31, stride=2):
        super(LargeConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.bn(self.conv(x)))

class AdaptiveNonStationarityEmbedding(nn.Module):
    """
    Section 3.3: Adaptive non-stationarity embedding block
    """
    def __init__(self, input_len, out_len, channels, kernel_size=15):
        super(AdaptiveNonStationarityEmbedding, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.padding)
        
        # Maps time dimension: T/8 -> T/32
        self.linear_trend = nn.Linear(input_len, out_len)
        self.linear_seasonal = nn.Linear(input_len, out_len)

    def forward(self, x_large):
        # x_large: [Batch, D, T/8]
        x_t = self.avg_pool(x_large)
        x_s = x_large - x_t
        
        # Linear projection on Time Dimension
        x_t_l = self.linear_trend(x_t)
        x_s_l = self.linear_seasonal(x_s)
        
        x_embedding = x_t_l + x_s_l
        return x_embedding

class MFABlock(nn.Module):
    """
    Section 3.4: Multi-scale fusion attention (MFA)
    """
    def __init__(self, dim):
        super(MFABlock, self).__init__()
        self.qkv_conv = nn.Conv1d(dim, dim * 3, kernel_size=1)
        self.project_out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [Batch, E, T/32]
        B, E, T = x.shape
        
        qkv = self.qkv_conv(x) # Enlarge_conv
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Horizontal & Vertical Pooling on K
        kx = torch.mean(k, dim=-1, keepdim=True) 
        ky = torch.mean(k, dim=1, keepdim=True)
        
        # First Interaction
        f = F.relu(q) * kx * ky
        
        # Pooling on F
        fx = torch.mean(f, dim=-1, keepdim=True)
        fy = torch.mean(f, dim=1, keepdim=True)
        
        # Second Interaction
        x_mfa = F.relu(v) * fx * fy
        
        # Linear Projection
        x_mfa = x_mfa.transpose(1, 2) # [B, T, E]
        x_mfa = self.project_out(x_mfa)
        x_mfa = x_mfa.transpose(1, 2) # [B, E, T]
        
        return x_mfa

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class NPFormer(nn.Module):
    def __init__(self, 
                 num_classes=10,
                 seq_len=1024,
                 in_channels=1,
                 d_model=16,   # D in paper
                 e_model=128,  # E in paper
                 large_kernel_size=31):
        super(NPFormer, self).__init__()
        
        self.in_channels = in_channels
        self.seq_len = seq_len

        # 1. Input AvgPooling
        self.input_avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # 2. Large Kernel Blocks
        self.large_block1 = LargeConvBlock(in_channels, d_model, kernel_size=large_kernel_size, stride=2)
        self.large_block2 = LargeConvBlock(d_model, d_model, kernel_size=large_kernel_size, stride=2)
        
        # Calculate intermediate lengths
        # Input T=1024 -> T/2=512 -> T/4=256 -> T/8=128
        self.l_8 = seq_len // 8
        self.l_32 = seq_len // 32
        
        # 3. Adaptive Non-stationarity Embedding
        self.adaptive_embedding = AdaptiveNonStationarityEmbedding(
            input_len=self.l_8, 
            out_len=self.l_32, 
            channels=d_model
        )
        
        # 4. Dimension Expansion
        self.expand_conv = nn.Conv1d(d_model, e_model, kernel_size=1)
        
        # 5. Transformer Encoder Layer (MFA + FFN + Norms)
        self.mfa_block = MFABlock(dim=e_model)
        self.norm1 = nn.LayerNorm(e_model)
        
        self.ffn = FeedForward(dim=e_model, hidden_dim=e_model * 4)
        self.norm2 = nn.LayerNorm(e_model)
        
        # 6. Output Head
        self.output_avg_pool = nn.AdaptiveAvgPool1d(1) 
        self.classifier = nn.Linear(e_model, num_classes)

    def forward(self, x):
        # Handle dimensions: Ensure input is [Batch, Channels, Length]
        # Some loaders in the project might provide [Batch, 1, 1, Length] or [Batch, Length]
        if x.dim() == 4:
            x = x.squeeze(1)  # [B, 1, 1, L] -> [B, 1, L]
        if x.dim() == 2:
            x = x.unsqueeze(1) # [B, L] -> [B, 1, L]
            
        # Step 3.2
        x = self.input_avg_pool(x)       # [B, C, T/2]
        x = self.large_block1(x)         # [B, D, T/4]
        x = self.large_block2(x)         # [B, D, T/8]
        
        # Step 3.3
        x = self.adaptive_embedding(x)   # [B, D, T/32]
        
        # Step 3.4 Preparation
        x_emb = self.expand_conv(x)      # [B, E, T/32]
        
        # MFA Attention with Residual & Norm
        x_mfa = self.mfa_block(x_emb)
        
        x_res1 = x_emb + x_mfa
        x_res1_t = x_res1.transpose(1, 2) # [B, T/32, E]
        x_norm1 = self.norm1(x_res1_t).transpose(1, 2)
        
        # FFN with Residual & Norm
        x_ffn = self.ffn(x_norm1)
        x_res2 = x_norm1 + x_ffn
        x_res2_t = x_res2.transpose(1, 2)
        x_attention = self.norm2(x_res2_t).transpose(1, 2) # [B, E, T/32]
        
        # Output
        y_avg = self.output_avg_pool(x_attention) # [B, E, 1]
        y_avg = y_avg.flatten(1)                  # [B, E]
        output = self.classifier(y_avg)           # [B, Num_Classes]
        
        # Return tuple (logits, features) to be compatible with train_op logic
        return output, y_avg

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