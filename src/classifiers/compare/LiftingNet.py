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

# This file contains the lifting scheme implementation
# There is no complete network definition inside this file.
# Note that it also contains other wavelet transformation
# used in WCNN and DAWN networks.

# To change if we do horizontal first inside the LS
HORIZONTAL_FIRST = True

class Splitting(nn.Module):
    def __init__(self, h=True):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction

        self.conv_even = lambda x: x[:, :, :, ::2]
        self.conv_odd = lambda x: x[:, :, :, 1::2]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))

class LiftingModule(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(LiftingModule, self).__init__()
        self.splitting = Splitting(h=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size =(1,kernel_size), stride = 1,
                               padding=0, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size =(1,kernel_size), stride = 1,
                               padding=0, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )
    def forward(self, x):
        # x [64,1,1,1024]
        (x_even, x_odd) = self.splitting(x)
        kernel_size = 3
        # predicting
        padding = x[:, :, :, -kernel_size//2:]
        x_even = torch.cat((x_even, padding), -1)
        x_even = self.conv1(x_even)
        x_odd = torch.cat((x_odd,x_odd,x_odd,x_odd,x_odd), 1)
        x_pre = x_odd - x_even
        # updating
        padding = x_pre[:, :, :, -kernel_size // 2:]
        x_pre = torch.cat((x_pre, padding), -1)
        x_pre = self.conv2(x_pre)
        x_upd = x_even + x_pre

        return x_upd

class LiftingNet(nn.Module):
    def __init__(self, in_planes, kernel_size, class_nums):
        super(LiftingNet, self).__init__()

        self.Lifting = nn.ModuleList()
        number_levels = 3
        out_planes = in_planes
        for i in range(number_levels):
            if i == 0:
                self.Lifting.add_module(
                    str(i),
                    LiftingModule(in_planes, in_planes*5, kernel_size)
                )
            else:
                self.Lifting.add_module(
                    str(i),
                    LiftingModule(out_planes, out_planes*5, kernel_size)
                )
            # Here you can change this number if you want compression
            out_planes = out_planes * 5
        out_planes = 125
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(out_planes, class_nums)
    def forward(self, x):

        for l in self.Lifting:
            x = l(x)
        x = self.maxpool(x)
        x = x.view(-1, x.size()[1])
        x = self.fc(x)

        return x, x

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