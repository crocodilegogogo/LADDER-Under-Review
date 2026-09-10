# encoding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
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

__all__ = ['MFSFormer']

# ------------------------------------------------------------------------------------------
# MFSFormer Model Components (Ref: TR2025 Paper)
# ------------------------------------------------------------------------------------------

class ChannelShuffle(nn.Module):
    """
    Channel Shuffle operation to facilitate information flow between groups.
    """
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, length = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batch_size, self.groups, channels_per_group, length)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, num_channels, length)
        return x

class MSCBlock(nn.Module):
    """
    Multiscale Convolution Block (MSC)
    Uses different kernel sizes to extract multiscale features.
    """
    def __init__(self, in_channels, out_channels):
        super(MSCBlock, self).__init__()
        # Split output channels among 3 branches to capture different scales
        c1 = out_channels // 4
        c2 = out_channels // 4
        c3 = out_channels - c1 - c2 
        
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(in_channels, c2, kernel_size=5, padding='same')
        self.conv3 = nn.Conv1d(in_channels, c3, kernel_size=7, padding='same')
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y = torch.cat([y1, y2, y3], dim=1)
        y = self.bn(y)
        y = self.act(y)
        return y

class FSABlock(nn.Module):
    """
    Fuse-Shuffle Attention (FSA) Block
    Captures dependencies between feature channels and windows.
    """
    def __init__(self, channels, groups=4):
        super(FSABlock, self).__init__()
        assert channels % groups == 0, "Channels must be divisible by groups"
        self.groups = groups
        
        # Sub-feature channels
        sub_channels = channels // groups
        self.half_sub = sub_channels // 2
        
        # Branch 1: Channel Attention (Global Info)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_channel = nn.Linear(self.half_sub, self.half_sub)
        
        # Branch 2: Window Attention (Local Info)
        self.gn = nn.GroupNorm(num_groups=1, num_channels=self.half_sub) 
        self.conv_window = nn.Conv1d(self.half_sub, self.half_sub, kernel_size=1)
        
        self.shuffle = ChannelShuffle(groups)

    def forward(self, x):
        B, C, L = x.size()
        x_reshaped = x.view(B, self.groups, -1, L)
        
        group_outputs = []
        for k in range(self.groups):
            sub_feature = x_reshaped[:, k, :, :]
            x_k1, x_k2 = torch.split(sub_feature, self.half_sub, dim=1)
            
            # Branch 1
            s      = self.gap(x_k1).squeeze(-1)
            w_s    = torch.sigmoid(self.fc_channel(s)).unsqueeze(-1)
            out_k1 = x_k1 * w_s
            
            # Branch 2
            gn_out = self.gn(x_k2)
            w_gn = torch.sigmoid(self.conv_window(gn_out))
            out_k2 = x_k2 * w_gn
            
            fused = torch.cat([out_k1, out_k2], dim=1)
            group_outputs.append(fused)
            
        y = torch.cat(group_outputs, dim=1)
        y = self.shuffle(y)
        return y

class MFSFormerLayer(nn.Module):
    """
    Single MFSFormer Encoder Layer.
    Structure: MSC -> FSA -> Norm&Add -> FFN -> Norm&Add
    """
    def __init__(self, d_model, groups=4, dropout=0.1):
        super(MFSFormerLayer, self).__init__()
        self.msc = MSCBlock(d_model, d_model)
        self.fsa = FSABlock(d_model, groups=groups)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Feature extraction via MSC & FSA
        msc_out = self.msc(x)
        fsa_out = self.fsa(msc_out)
        
        # Skip connection 1
        x = self.norm1(x + self.dropout(fsa_out))
        
        # FFN & Skip connection 2
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class MFSFormer(nn.Module):
    """
    MFSFormer Model Architecture
    """
    def __init__(self, num_classes, d_model=64, num_layers=3, groups=4, input_len=1024):
        super(MFSFormer, self).__init__()
        
        # Input Layer: AvgPool -> Conv -> BN -> GELU
        self.input_pool = nn.AvgPool1d(kernel_size=2, stride=2) 
        self.input_conv = nn.Conv1d(1, d_model, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(d_model)
        self.input_act = nn.GELU()
        
        # Encoder Layers
        self.layers = nn.ModuleList([
            MFSFormerLayer(d_model, groups=groups) for _ in range(num_layers)
        ])
        
        # Output Layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Ensure input is (Batch, Channel, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            x = x.squeeze(2) # Change to (B, 1, L)
            
        # Input processing
        x = self.input_pool(x)
        x = self.input_conv(x)
        x = self.input_act(self.input_bn(x))
        
        # Encoder processing
        for layer in self.layers:
            x = layer(x)
            
        # Classification
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        
        # Return logits twice to match the project's ResNet signature (logits, features/logits)
        # The project train_op uses output[0] for loss
        return logits, x

# ------------------------------------------------------------------------------------------
# Training and Evaluation Functions (Copied from Resnet18.py for compatibility)
# ------------------------------------------------------------------------------------------

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