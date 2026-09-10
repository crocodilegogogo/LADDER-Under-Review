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
from utils.DWT_IDWT.DWT_IDWT_layer import *
import os

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
import torch.nn as nn


# device = '1'
# class soft_thresshold(nn.Module):
#     def __init__(self, input_feature, k = 1):
#         super(soft_thresshold, self).__init__()
#         self.average = nn.AdaptiveAvgPool2d(1)
#         self.FC1 = nn.Linear( input_feature, k * input_feature)
#         self.BN = nn.BatchNorm1d( k * input_feature)
#         self.relu = nn.ReLU()
#         self.FC2 = nn.Linear(k * input_feature, input_feature)

#     def forward(self, x):
#         bath_size = x.shape[0]
#         feature_size = x.shape[1]
#         lenth = x.shape[-1]

#         average = self.average(x)
#         average = average.view(bath_size, feature_size)
#         scales = self.FC1(average)
#         scales = self.BN(scales)
#         scales = self.relu(scales)
#         scales = self.FC2(scales)
#         scales = torch.sigmoid(scales).to(device)
#         thre = torch.multiply(average, scales)

#         thre = thre.view(bath_size, feature_size, 1, 1).to(device)
#         thres = (torch.abs(x) - thre).view(-1).to(device)
#         y = torch.sign(x).view(-1).to(device)
#         zeros = torch.zeros((bath_size, feature_size, 1, 200)).view(-1).to(device)
#         threshold = torch.multiply(y, torch.maximum(thres, zeros)).to(device)
#         threshold = threshold.view(bath_size, feature_size, 1 , 200).to(device)
#         return threshold

# device = '1'
class PositionalEncoding(nn.Module):
    #  self.position_encode = PositionalEncoding(feature_channel_out, drop_rate2, data_length)
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe = pe.transpose(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # x = x + Variable(self.pe, requires_grad=False)
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8, drop_rate=0):
        super(SelfAttention, self).__init__()
        self.k, self.heads = k, heads
        
        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)
        
        self.dropout_attention = nn.Dropout(drop_rate)
        
        self.unifyheads = nn.Linear(heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys = self.tokeys(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)
        
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)
        # normalization
        queries = queries / (k ** (1 / 4))
        keys = keys / (k ** (1 / 4))
        
        dot = torch.bmm(queries, keys.transpose(1, 2))
        
        dot = F.softmax(dot, dim=2)
        dot = self.dropout_attention(dot)
        out = torch.bmm(dot, values).view(b, h, t, k)
        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unifyheads(out)  # (b, t, k)

class Downsample_L(nn.Module):
    def __init__(self, wavename = 'db'):
        super(Downsample_L, self).__init__()
        self.dwt = DWT_1D_L(wavename = wavename)

    def forward(self, input):
        L = self.dwt(input)
        return L

class Downsample(nn.Module):
    def __init__(self, wavename = 'db'):
        super(Downsample, self).__init__()
        self.dwt = DWT_1D(wavename = wavename)

    def forward(self, input):
        L,H = self.dwt(input)
        return L,H

class Upsample(nn.Module):
    def __init__(self, wavename = 'db'):
        super(Upsample, self).__init__()
        self.idwt = IDWT_1D(wavename = wavename)

    def forward(self, L, H):
        out = self.idwt(L, H)
        return out

class FuseAttention(nn.Module):
    def __init__(self, k, wavename, heads=8, drop_rate=0):
        super(FuseAttention, self).__init__()
        self.k, self.heads = k, heads
        
        self.tokeys = nn.Linear(k, k * heads, bias=False)
        self.toqueries = nn.Linear(k, k * heads, bias=False)
        self.toqueries_up = nn.Linear(k, 2* k * heads, bias=False)
        self.tovalues = nn.Linear(k, k * heads, bias=False)
        
        self.dropout_attention = nn.Dropout(drop_rate)
        
        self.idwt =  IDWT_1D(wavename = wavename)

        self.transition = nn.Sequential(
            nn.Conv2d(k * 2, k, 1, 1),
            nn.BatchNorm2d(k),
            nn.LeakyReLU(),
        )

    def forward(self, x, y, data_length):
        b, t_x, k = x.size()
        b, t_y, k = y.size()
        h = self.heads
        queries = self.toqueries(y).view(b, t_y, h, k)
        keys = self.tokeys(x).view(b, t_x, h, k)
        values = self.tovalues(x).view(b, t_x, h, k)
        
        queries = queries.transpose(1, 2).contiguous().view(b * h, t_y, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t_x, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t_x, k)
        # normalization
        queries = queries
        keys = keys / (k ** (1 / 2))
        
        dot = torch.bmm(queries, keys.transpose(1, 2)).transpose(1, 2)
        
        dot = F.softmax(dot, dim=1)
        dot = self.dropout_attention(dot)
        out1 = torch.mul(dot[:,0,:], values[:,0,:]).unsqueeze(dim=1).view(b, h, t_y, k)
        out2 = torch.mul(dot[:,1,:], values[:,1,:]).unsqueeze(dim=1).view(b, h, t_y, k)
        out3 = torch.mul(dot[:,2,:], values[:,2,:]).unsqueeze(dim=1).view(b, h, t_y, k)
        out4 = torch.mul(dot[:,3,:], values[:,3,:]).unsqueeze(dim=1).view(b, h, t_y, k)

        # swap h, t back, unify heads
        out1 = out1.transpose(1, 2).contiguous().view(b, t_y, h * k)
        out2 = out2.transpose(1, 2).contiguous().view(b, t_y, h * k)
        out3 = out3.transpose(1, 2).contiguous().view(b, t_y, h * k)
        out4 = out4.transpose(1, 2).contiguous().view(b, t_y, h * k)

        # [b, f, c, l]
        out1 = out1.reshape(int(b/data_length), data_length, 1,  k).permute(0, 3, 2, 1)
        out2 = out2.reshape(int(b/data_length), data_length, 1,  k).permute(0, 3, 2, 1)
        out3 = out3.reshape(int(b/data_length), data_length, 1,  k).permute(0, 3, 2, 1)
        out4 = out4.reshape(int(b/data_length), data_length, 1,  k).permute(0, 3, 2, 1)

        out_L = self.idwt(out1, out2)
        out_H = self.idwt(out3, out4)

        out = torch.cat((out_L, out_H), 1)
        out =  self.transition(out)

        return out  # (b, t, k)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads, drop_rate):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(k, heads=heads, drop_rate=drop_rate)
        # self.norm1 = nn.LayerNorm(k)
        self.norm1 = nn.BatchNorm1d(k)
        self.mlp = nn.Sequential(
            nn.Conv1d(k, 4 * k, 1, 1),
            nn.ReLU(),
            nn.Conv1d(4 * k, k, 1, 1)
        )

        self.norm2 = nn.BatchNorm1d(k)

        self.dropout_forward = nn.Dropout(drop_rate)

    def forward(self, x):
        # self-attention
        attended = self.attention(x)
        attended = attended + x
        attended = attended.permute(0,2,1)

        # layer norm
        x = self.norm1(attended)
        # feedforward and layer norm
        feedforward = self.mlp(x)

        feedforward = feedforward + x

        return self.dropout_forward(self.norm2(feedforward).permute(0,2,1))


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class _4dwt_4(nn.Module):
    def __init__(self, input_2Dfeature_channel, kernel_size,
                 feature_channel, feature_channel_out, drop_rate, num_class, multiheads=1, wavename = "db4"):
        super(_4dwt_4, self).__init__()

        ################################# 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_2Dfeature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),

        )
        self.dwt = nn.Sequential(Downsample(wavename=wavename),)
        self.idwt = IDWT_1D(wavename=wavename)
        self.dwt_L = nn.Sequential(Downsample_L(wavename=wavename),)
        self.Attention_Fusion = FuseAttention(feature_channel, wavename,multiheads, drop_rate)

        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )

        ################################# 2

        self.Attention_Fusion1 =  FuseAttention(feature_channel, wavename,multiheads, drop_rate)

        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )

        ################################# 2

        self.Attention_Fusion2 =  FuseAttention(feature_channel, wavename,multiheads, drop_rate)

        self.conv5 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )
        ################################# 2

        self.Attention_Fusion3 =  FuseAttention(feature_channel, wavename,multiheads, drop_rate)

        self.conv7 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1, kernel_size), 1, (0, kernel_size // 2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
        )

        self.transformer_block1 = TransformerBlock(feature_channel_out, multiheads, drop_rate)

        self.transformer_block2 = TransformerBlock(feature_channel_out, multiheads, drop_rate)

        self.global_ave_pooling = nn.AdaptiveAvgPool1d(1)

        self.linear = nn.Linear(feature_channel_out, num_class)

    def forward(self, x):
        
        x = self.conv1(x)  # [64, 8, 1, 512]
        batch_size = x.shape[0]
        input_channel = x.shape[2]
        feature_channel = x.shape[1]
        data_length = x.shape[-1]
        
        # decomposition
        x_L, x_H = self.dwt(x)  # [64, 8, 1, 256]
        x_LL, x_LH = self.dwt(x_L)
        x_HL, x_HH = self.dwt(x_H) # [64, 8, 1, 128]

        x_all = torch.cat((x_LL, x_LH, x_HL, x_HH), dim=2)  # [64, 8, 2, 64]
        x_all = x_all.permute((0, 3, 2, 1)).reshape(-1, 4, feature_channel) # [64, 64, 16]

        # low frequency path
        x_L = self.conv2(x_L)  # [64, 8, 1, 256]
        x_LL0 = self.dwt_L(x_L)  # [64, 8, 1, 128]
        x_LL0 = x_LL0.permute(0, 3, 2, 1).reshape(-1, 1, feature_channel)  # [64*128, 1, 8]

        # fusion
        data_length = data_length // 4 # 128
        x = self.Attention_Fusion(x_all, x_LL0, data_length)  # [64, 8, 1, 256]
        x = x + x_L

        ########################## 2 ################################
        x = self.conv3(x)  # [64, 8, 1, 256]

        # decomposition
        x_L, x_H = self.dwt(x)
        x_LL, x_LH = self.dwt(x_L)
        x_HL, x_HH = self.dwt(x_H) # [64, 8, 1, 64]

        x_all = torch.cat((x_LL, x_LH, x_HL, x_HH), dim=2)  # [64, 8, 2, 64]
        x_all = x_all.permute((0, 3, 2, 1)).reshape(-1, 4, feature_channel) # [64, 64, 16]

        # low frequency path
        x_L = self.conv4(x_L)  # [64, 8, 1, 256]
        x_LL0 = self.dwt_L(x_L)  # [64, 8, 1, 128]
        x_LL0 = x_LL0.permute(0, 3, 2, 1).reshape(-1, 1, feature_channel)  # [64*128, 1, 8]

        # fusion
        data_length = data_length // 2 # 64
        x = self.Attention_Fusion1(x_all, x_LL0, data_length)  # [64*128, 1, 8]
        x = x + x_L
        ########################## 2 ################################

        x = self.conv5(x)  # [64, 8, 1, 256]

        # decomposition
        x_L, x_H = self.dwt(x)
        x_LL, x_LH = self.dwt(x_L)
        x_HL, x_HH = self.dwt(x_H)  # [64, 8, 1, 64]

        x_all = torch.cat((x_LL, x_LH, x_HL, x_HH), dim=2)  # [64, 8, 2, 64]
        x_all = x_all.permute((0, 3, 2, 1)).reshape(-1, 4, feature_channel)  # [64, 64, 16]

        # low frequency path
        x_L = self.conv6(x_L)  # [64, 8, 1, 256]
        x_LL0 = self.dwt_L(x_L)  # [64, 8, 1, 128]
        x_LL0 = x_LL0.permute(0, 3, 2, 1).reshape(-1, 1, feature_channel)  # [64*128, 1, 8]

        # fusion
        data_length = data_length // 2  # 64
        x = self.Attention_Fusion2(x_all, x_LL0, data_length)  # [64*128, 1, 8]
        x = x + x_L
        ########################## 2 ################################

        x = self.conv7(x)  # [64, 8, 1, 256]

        # decomposition
        x_L, x_H = self.dwt(x)
        x_LL, x_LH = self.dwt(x_L)
        x_HL, x_HH = self.dwt(x_H)  # [64, 8, 1, 64]

        x_all = torch.cat((x_LL, x_LH, x_HL, x_HH), dim=2)  # [64, 8, 2, 64]
        x_all = x_all.permute((0, 3, 2, 1)).reshape(-1, 4, feature_channel)  # [64, 64, 16]

        # low frequency path
        x_L = self.conv8(x_L)  # [64, 8, 1, 256]
        x_LL0 = self.dwt_L(x_L)  # [64, 8, 1, 128]
        x_LL0 = x_LL0.permute(0, 3, 2, 1).reshape(-1, 1, feature_channel)  # [64*128, 1, 8]

        # fusion
        data_length = data_length // 2  # 64
        x = self.Attention_Fusion3(x_all, x_LL0, data_length)  # [64*128, 1, 8]
        x = x + x_L  # [64, 48, 1, 128]
        #############################
        data_length = x.shape[-1]
        # x = self.transition(x)

        x = x.view(batch_size, -1, data_length) # [64,48,64]

        x = x.permute(0, 2, 1)

        x = self.transformer_block1(x)
        x = self.transformer_block2(x)  #torch.Size([64, 256, 32])

        x = x.permute(0, 2, 1)

        x = self.global_ave_pooling(x).squeeze()

        output = self.linear(x)

        return output, x

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