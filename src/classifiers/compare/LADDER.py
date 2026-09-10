import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import time
from utils.utils import *
import os

# To change if we do horizontal first inside the LS
HORIZONTAL_FIRST = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# split the even and odd timestamps
class Splitting(nn.Module):
    def __init__(self, h=True):
        super(Splitting, self).__init__()
        # Deciding the stride base on the direction

        self.conv_even = lambda x: x[:, :, :, ::2]
        self.conv_odd = lambda x: x[:, :, :, 1::2]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.conv_even(x), self.conv_odd(x))

# the lifting wavelet decomposition module
class LiftingScheme(nn.Module):
    def __init__(self, i, in_planes, modified=True, splitting_flag=True, k_size=3, simple_lifting=False):
        super(LiftingScheme, self).__init__()
        self.modified = modified

        kernel_size = (1, k_size)
        pad = (k_size // 2, k_size - 1 - k_size // 2, 0, 0)

        self.splitting_flag = splitting_flag
        self.split = Splitting(h=True)

        # Dynamic build sequential network
        modules_P = []
        modules_U = []

        # HARD CODED Architecture
        if simple_lifting:
            modules_P = modules_P + [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1, bias=False),
                nn.Tanh()
            ]
            modules_U = modules_U + [
                nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1, bias=False),
                nn.Tanh()
            ]
        else:
            modules_P = modules_P + [
                # nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=kernel_size, stride=1, bias=False, dilation=(1, math.ceil(50/2**(i+1))), padding = (0, math.ceil(50/2**(i+1)))),
                nn.PReLU()
            ]
            modules_P = modules_P + [
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=(1, 1), stride=1, bias=False),
                nn.Tanh()
            ]
            
            modules_U = modules_U + [
                # nn.ReflectionPad2d(pad),
                nn.Conv2d(in_planes , in_planes,
                          kernel_size=kernel_size, stride=1, bias=False, dilation=(1, math.ceil(50/2**(i+1))), padding = (0, math.ceil(50/2**(i+1)))),
                nn.PReLU()
            ]
            modules_U = modules_U + [
                nn.Conv2d(in_planes, in_planes,
                          kernel_size=(1, 1), stride=1, bias=False),
                nn.Tanh()
            ]
        
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting_flag:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        if self.modified:
            c = x_even + self.U(x_odd)
            d = x_odd - self.P(c)
            return c, d
        else:
            d = x_odd - self.P(x_even)
            c = x_even + self.U(d)
            return c, d

class LevelDAWN(nn.Module):
    def __init__(self, dataset_name, i, in_planes, kernel_size, no_bottleneck,
                 share_weights, simple_lifting, regu_details, regu_approx):
        super(LevelDAWN, self).__init__()
        self.regu_details = regu_details
        self.regu_approx  = regu_approx
        self.dataset_name = dataset_name
        if self.regu_approx + self.regu_details > 0.0:
            self.loss_details = nn.SmoothL1Loss()

        self.wavelet = LiftingScheme(i, in_planes, modified=True, splitting_flag=True,
                                     k_size=kernel_size, simple_lifting=simple_lifting)
        self.share_weights = share_weights

    def forward(self, x):

        L, H = self.wavelet(x)

        rd = self.regu_details * H.abs().mean()
        rc = self.regu_approx  * torch.dist(L.mean(), x.mean(), p=2) # calculate the distance between L.mean() and x.mean()
        r  = rd + rc

        return L, H, r

def gumbel_softmax(x, dim, tau):
    x = torch.softmax(x, dim=1)
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class gumble_block_2D(nn.Module):
    def __init__(self, first_conv_chnnl, inchannel, outchannel):
        super(gumble_block_2D, self).__init__()
        self.ch_mask_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannel, inchannel, kernel_size=1, bias=False),
            nn.PReLU(),
            nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
            nn.PReLU()
        )

        self.tau = 1  # nn.Parameter(torch.tensor([1.]))
        self.inchannel  = inchannel
        self.outchannel = outchannel

    def _update_tau(self, tau):
        self.tau = tau

    def forward(self, x_low, x_high, i, ch_mask, test_flag):
        
        x   = torch.cat((x_low.unsqueeze(1), x_high.unsqueeze(1)), dim=1)
        out = x 
        x   = x.reshape(-1, self.inchannel, 1, x.shape[4])
        ch_mask_1 = self.ch_mask_1(x)

        ch_mask_1 = gumbel_softmax(ch_mask_1, dim=1, tau=self.tau)
        ch_mask_1 = ch_mask_1.reshape(x_low.shape[0], -1, self.outchannel, 1, 1)

        if test_flag == False:
            if i == 0:
                input_conv  = out * (ch_mask_1[:,:,0,:,:].unsqueeze(-1))
                input_res  = out * (ch_mask_1[:,:,1,:,:].unsqueeze(-1))
            else:
                input_conv  = out * (ch_mask_1[:,:,0,:,:].unsqueeze(2))
                input_res   = out * (ch_mask_1[:,:,1,:,:].unsqueeze(2))
        if test_flag == True:
            ##################################################
            ch_mask_1 = torch.argmax(ch_mask_1, 2).unsqueeze(-1)
            ch_mask_1 = ch_mask_1.contiguous().view(-1).unsqueeze(-1)
            ##################################################
            onehot_ch_mask_1 = torch.zeros([ch_mask_1.shape[0], 2], device='cuda')
            onehot_ch_mask_1.scatter_(1, ch_mask_1, 1) # [16*2,2]
            ch_mask_1 = onehot_ch_mask_1.reshape(out.shape[0], 2, 2).unsqueeze(-1).unsqueeze(-1)
            
            if i != 0:
                # mul previous masks
                prev_mask = ch_mask[i-1][0]
                ch_mask_1[:,:,0,:,:] = (ch_mask_1[:,:,0,:,:].unsqueeze(2) * prev_mask).squeeze(2)

            input_conv = out * (ch_mask_1[:,:,0,:,:].unsqueeze(-1))
            input_res  = out * (ch_mask_1[:,:,1,:,:].unsqueeze(-1))

        return input_conv, input_res, ch_mask_1[:,:,0,:,:].reshape(-1,1,1,1,1)

class gumble_block_2D_all(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(gumble_block_2D_all, self).__init__()
        self.ch_mask_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannel, inchannel, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=True),
            nn.PReLU()
        )

        self.tau1 = 1
        self.outchannel = outchannel

    def _update_tau1(self, tau1):
        self.tau1 = tau1

    def forward(self, x, test_flag):

        out = x
        ch_mask_1 = self.ch_mask_1(x)

        ch_mask_1 = gumbel_softmax(ch_mask_1, dim=1, tau=self.tau1).unsqueeze(-1)

        out = out.view(out.shape[0], self.outchannel, -1, out.shape[2], out.shape[3])

        if test_flag == True:
            ch_mask_1 = torch.argmax(ch_mask_1, 1).squeeze(-1).squeeze(-1)
            onehot_ch_mask_1 = torch.FloatTensor(out.shape[0], out.shape[1]).to(out.device)
            onehot_ch_mask_1.zero_()
            onehot_ch_mask_1.scatter_(1, ch_mask_1, 1)
            ch_mask_1 = onehot_ch_mask_1.reshape(out.shape[0], -1, 1, 1, 1)

        input_conv = torch.sum(out * ch_mask_1, dim=1)
        return input_conv, ch_mask_1

class gumble_block_2D_all_pooling(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(gumble_block_2D_all_pooling, self).__init__()
        self.ch_mask_1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(inchannel, outchannel, kernel_size=1),
            nn.PReLU()
        )
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.tau1 = 1
        self.outchannel = outchannel

    def _update_tau1(self, tau1):
        self.tau1 = tau1

    def forward(self, x, test_flag):

        out = x
        x         = self.avgpool(x.permute(0,1,5,3,4,2))
        x         = x.reshape(x.shape[0],-1,1,1)
        ch_mask_1 = self.ch_mask_1(x)

        ch_mask_1 = gumbel_softmax(ch_mask_1, dim=1, tau=self.tau1).unsqueeze(-1)

        out = out.view(out.shape[0], self.outchannel, -1, out.shape[2], out.shape[3])

        if test_flag == True:
            ch_mask_1 = torch.argmax(ch_mask_1, 1).squeeze(-1).squeeze(-1)
            onehot_ch_mask_1 = torch.FloatTensor(out.shape[0], out.shape[1]).to(out.device)
            onehot_ch_mask_1.zero_()
            onehot_ch_mask_1.scatter_(1, ch_mask_1, 1)
            ch_mask_1 = onehot_ch_mask_1.reshape(out.shape[0], -1, 1, 1, 1)

        input_conv = torch.sum(out * ch_mask_1, dim=1)
        return input_conv, ch_mask_1

def reshape_gumbel_features(x, o, batch_size, ch_mask, i, test_flag):
    
    x         = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
    x_current = x.reshape(batch_size, -1, x.shape[1], x.shape[2], x.shape[3]).permute(0,2,3,1,4)
    x_current = x_current.reshape(batch_size, x.shape[1], 1, -1)
    o         = o.reshape(-1, o.shape[2], o.shape[3], o.shape[4])
    o_current = o.reshape(batch_size, -1, o.shape[1], o.shape[2], o.shape[3]).permute(0,2,3,1,4)
    o_current = o_current.reshape(batch_size, o.shape[1], 1, -1)
    return x_current, o_current, x, o

# main
class LADDER(nn.Module):
    def __init__(self, dataset_name, num_classes, data_lenth, first_conv_chnnl=16,
                 kernel_size=3, no_bootleneck=False, average_mode="mode2",
                 classifier="mode2", share_weights=False, simple_lifting=False,
                 COLOR=True, regu_details=0.01, regu_approx=0.01, haar_wavelet=False):
        super(LADDER, self).__init__()
        
        number_decomp_levels      = 8 # find_power(data_lenth)-2 # set decomposition levels
        self.number_decomp_levels = number_decomp_levels
        self.data_lenth           = data_lenth
        self.average_mode         = average_mode
        self.dataset_name         = dataset_name
        
        # self.conv_in large_kernel_sized conv for feature extraction
        if dataset_name in ['SEU','IMS','PU']:
            self.conv_in = nn.Sequential(
                nn.Conv2d(1, first_conv_chnnl*4,
                          kernel_size=(1, kernel_size+100), stride=1, padding=(0, (kernel_size+100) // 2)),
                nn.BatchNorm2d(first_conv_chnnl*4),
                nn.PReLU(),
                nn.Conv2d(first_conv_chnnl*4, first_conv_chnnl*4,
                          kernel_size=(1, (kernel_size+2)), stride=1,
                          padding= (0, (kernel_size+2) // 2)),
                nn.BatchNorm2d(first_conv_chnnl*4),
                nn.PReLU(),
                nn.Conv2d(first_conv_chnnl*4, first_conv_chnnl,
                          kernel_size=(1, (kernel_size+2)), stride=1, 
                          padding= (0, (kernel_size+2) // 2)),
                nn.BatchNorm2d(first_conv_chnnl),
                nn.PReLU()
            )
        else:
            self.conv_in = nn.Sequential(
                nn.Conv2d(1, first_conv_chnnl*4,
                          kernel_size=(1, kernel_size+100), stride=1, padding=(0, (kernel_size+100) // 2)),
                nn.BatchNorm2d(first_conv_chnnl*4),
                nn.PReLU(),
                nn.Conv2d(first_conv_chnnl*4, first_conv_chnnl,
                          kernel_size=(1, (kernel_size+2)), stride=1,
                          padding= (0, (kernel_size+2) // 2)),
                nn.BatchNorm2d(first_conv_chnnl),
                nn.PReLU()
            )

        # Initialize the recording list of different levels sequentially
        self.rs      = []  # List of constrains on details and mean
        self.feature_layer = []
        self.ch_mask = []
        self.feature_all_levels = []
        for i in range(self.number_decomp_levels):
            # rs.append([])
            self.feature_layer.append([]) # save the features of each decomposition level
            self.ch_mask.append([])

        # decomposition and selection of freq-bands and termination-level
        self.decomp_levels    = nn.ModuleList() # save the fre-band decomposition modules of each decomposition level
        self.gumble_blocks    = nn.ModuleList() # save the fre-band selection modules for further decomposition
        self.convs_o          = nn.ModuleList()
        in_planes  = first_conv_chnnl
        out_planes = first_conv_chnnl
        # the fre-band decomposition and selection modules of different decomposition levels
        for i in range(number_decomp_levels):
            nobootleneck = True
            if no_bootleneck and i == number_decomp_levels - 1:
                nobootleneck = False
            # the fre-band decomposition modules
            self.decomp_levels.add_module(
                str(i),
                LevelDAWN(self.dataset_name, i, in_planes,
                          kernel_size, nobootleneck,
                          share_weights, simple_lifting, regu_details, regu_approx),
            )
            # the fre-band selection modules
            self.gumble_blocks.add_module(
                str(i),
                gumble_block_2D(first_conv_chnnl, in_planes, 2)
            )
            self.convs_o.add_module(
                    str(i),
                    nn.Sequential(nn.Conv2d(first_conv_chnnl, first_conv_chnnl,
                              kernel_size=(1, 3), stride=1,
                              dilation=(1, math.ceil(50/2**(i+1))), padding = (0, math.ceil(50/2**(i+1))), bias=False),
                    nn.PReLU()),
                    )
            
        # Adaptive selection of wavelet decomposition levels
        self.gumble_blocks_all = gumble_block_2D_all(in_planes * number_decomp_levels, number_decomp_levels)
        
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        
        if average_mode == "mode1":
            self.conv_out = nn.Sequential(
                nn.Conv2d(first_conv_chnnl, 8*first_conv_chnnl,
                          kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2)),
                nn.BatchNorm2d(8*first_conv_chnnl),
                nn.PReLU()
            )
            # Time attention
            self.time_attention = nn.Sequential(
                nn.Conv2d(8*first_conv_chnnl, 4*first_conv_chnnl,
                          kernel_size=(1, 1), stride=1, padding=(0, 1 // 2)),
                nn.BatchNorm2d(4*first_conv_chnnl),
                nn.PReLU(),
                nn.Conv2d(4*first_conv_chnnl, 1,
                          kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2)),
                nn.Tanh()
                )
        
        if average_mode == "mode3":
            # Adaptive selection of wavelet decomposition levels
            self.gumble_blocks_all = gumble_block_2D_all_pooling(in_planes * number_decomp_levels, number_decomp_levels)

        # Classifier
        if classifier == "mode1":
            self.fc = nn.Linear(8*first_conv_chnnl, num_classes)
        elif classifier == "mode2":
            # Add one more layer in the classifier!
            self.fc = nn.Sequential(
                nn.Linear((number_decomp_levels+1)*first_conv_chnnl, (number_decomp_levels+1)*first_conv_chnnl),
                nn.BatchNorm1d((number_decomp_levels+1)*first_conv_chnnl),
                nn.PReLU(),
                nn.Linear((number_decomp_levels+1)*first_conv_chnnl, num_classes)
            )
        else:
            raise "Unknown classifier"

    def forward(self, x, test_flag=False):
        
        batch_size  = x.shape[0]
        
        ###########################
        x = self.conv_in(x)
        # Initialize the recording list
        rs            = self.rs.copy()
        feature_all_levels = self.feature_all_levels.copy()
        ch_mask       = self.ch_mask.copy()
        pooling_feature_all_levels_x = torch.zeros([batch_size, self.number_decomp_levels, self.number_decomp_levels+1,
                                                   x.shape[1], 1, 1], device='cuda')
        pooling_feature_all_levels_o = torch.zeros([batch_size, self.number_decomp_levels, self.number_decomp_levels+1,
                                                   x.shape[1], 1, 1], device='cuda')
        
        # adaptive wavelet decomposition routing
        for i in range(self.number_decomp_levels):
            if i == 0:
                x, H, r         = self.decomp_levels[i](x) # wavelet decomposition module
                x, o, ch_mask_i = self.gumble_blocks[i](x, H, i, ch_mask, test_flag) # frequency band selection module, x: selected fre-band(s)
                # Add the constrain of this level
                rs = rs + [r]
                
                x_current, o_current, x, o = reshape_gumbel_features(x,o,batch_size,ch_mask,i,test_flag)
                # Original
                o_current  = self.convs_o[i](o_current)
                # record the band not for decomposition of level-0
                no_decomp_features = o_current
                # record the features of all levels
                feature_all_levels = feature_all_levels + [o_current + x_current]
                # record the gumbel masks
                ch_mask[i]         = ch_mask[i] + [ch_mask_i]
                if self.average_mode == "mode2" or self.average_mode == "mode3":
                    pooling_feature_all_levels_x[:,i,i+1,:,:,:] = self.avgpool(x_current)
                    pooling_feature_all_levels_o[:,i,i,:,:,:] = self.avgpool(o_current)
                
            else:
                x, H, r         = self.decomp_levels[i](x)
                x, o, ch_mask_i = self.gumble_blocks[i](x, H, i, ch_mask, test_flag)
                rs = rs + [r]
                
                x_current, o_current, x, o = reshape_gumbel_features(x,o,batch_size,ch_mask,i,test_flag)
                # Original
                o_current = self.convs_o[i](o_current)
                # record and sum the band not for decomposition of i-th level
                no_decomp_features = no_decomp_features + o_current
                # record the features of all levels
                feature_all_levels = feature_all_levels + [no_decomp_features + x_current]
                # record the gumbel masks
                ch_mask[i]         = ch_mask[i] + [ch_mask_i]
                if self.average_mode == "mode2" or self.average_mode == "mode3":
                    pooling_feature_all_levels_x[:,i,i+1,:,:,:] = self.avgpool(x_current)
                    pooling_feature_all_levels_o[:,i,i,:,:,:]   = self.avgpool(o_current)
                    pooling_feature_all_levels_o[:,i,:,:,:,:]   = pooling_feature_all_levels_o[:,i-1,:,:,:,:] + pooling_feature_all_levels_o[:,i,:,:,:,:]
        
        
        if self.average_mode == "mode1":
            x                      = torch.cat(feature_all_levels, 1)
            # adaptive wavelet decomposition level selection
            x, ch_mask_level       = self.gumble_blocks_all(x, test_flag)
            x                      = self.conv_out(x)
            time_attns             = self.time_attention(x)
            x                      = x * F.softmax(time_attns, dim=-1)
            x                      = self.avgpool(x)
        elif self.average_mode == "mode2":
            x                      = torch.cat(feature_all_levels, 1)
            # adaptive wavelet decomposition level selection
            _, ch_mask_level             = self.gumble_blocks_all(x, test_flag) # ch_mask_level:[16, 8, 1, 1, 1]
            pooling_feature_all_levels_x = pooling_feature_all_levels_o + pooling_feature_all_levels_x
            pooling_feature_all_levels_x = pooling_feature_all_levels_x.reshape(batch_size,self.number_decomp_levels,-1,1,1) # [16, 8, 9*16, 1, 1]
            x                            = torch.sum((pooling_feature_all_levels_x * ch_mask_level), dim=1).squeeze(-1).squeeze(-1)
        elif self.average_mode == "mode3":
            pooling_feature_all_levels_x = pooling_feature_all_levels_o + pooling_feature_all_levels_x
            # adaptive wavelet decomposition level selection
            x, ch_mask_level             = self.gumble_blocks_all(pooling_feature_all_levels_x, test_flag) #[16,9*16,1,1]
            x                            = x.squeeze(1).reshape(batch_size,-1)

        x = x.reshape(-1, x.shape[1])
        x = self.fc(x)

        return x, rs

def train_op(network, EPOCH, BATCH_SIZE, LR,
             train_x, train_y, val_x, val_y,
             output_directory_models, log_training_duration, test_split):
    test_flag = False
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
    loss_class_train_results = []
    loss_class_val_results = []
    loss_class_test_results = []
    loss_regu_train_results = []
    loss_regu_val_results = []
    loss_regu_test_results = []

    loss_train_results = []
    accuracy_train_results = []
    loss_validation_results = []
    accuracy_validation_results = []
    loss_test_results = []
    accuracy_test_results = []

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

        epoch_tau = epoch + 1
        tau  = max((1 - (epoch_tau) / 250), 0.3)
        tau1 = max((1 - (epoch_tau) / 250), 0.3)
        for m in network.modules():
            if hasattr(m, '_update_tau'):
                m._update_tau(tau)

        for m in network.modules():
            if hasattr(m, '_update_tau1'):
                m._update_tau1(tau1)
                # print(a)

        for step, (x, y) in enumerate(train_loader):
            # h_state = None      # for initial hidden state

            batch_x = x.cuda()
            batch_y = y.cuda()
            # output_bc = network(batch_x)[0]
            output_bc, regus = network(batch_x)
            # cal the sum of pre loss per batch
            loss_class = loss_function(output_bc, batch_y)
            ### arguement
            loss_total = loss_class
            # If no regularisation used, None inside regus
            if regus[0]:
                loss_regu = sum(regus)
                loss_total = loss_total + loss_regu
            ###
            optimizer.zero_grad()
            loss_total.backward(retain_graph=True)
            optimizer.step()

        # test per epoch
        network.eval()
        # loss_train: loss of training set; accuracy_train: pre acc of training set
        # loss, acc, loss_regu, loss_class
        loss_train, accuracy_train, loss_class_train, loss_regu_train\
            = get_test_loss_acc_lifting_gumbel(network, loss_function, train_x, train_y, test_flag, test_split)
        test_flag = True
        loss_validation, accuracy_validation, loss_class_val, loss_regu_val\
            = get_test_loss_acc_lifting_gumbel(network, loss_function, val_x, val_y, test_flag, test_split)
        network.train()
        # update lr
        scheduler.step(loss_train)
        lr = optimizer.param_groups[0]['lr']
        # log lr&train&validation loss&acc per epoch
        lr_results.append(lr)
        loss_train_results.append(loss_train)
        accuracy_train_results.append(accuracy_train)
        loss_validation_results.append(loss_validation)
        accuracy_validation_results.append(accuracy_validation)

        loss_class_train_results.append(loss_class_train)
        loss_regu_train_results.append(loss_regu_train)
        loss_class_val_results.append(loss_class_val)
        loss_regu_val_results.append(loss_regu_val)
        
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
    history = log_history_all(EPOCH, lr_results, loss_train_results, accuracy_train_results,
                loss_validation_results, accuracy_validation_results,
                loss_test_results, accuracy_test_results,
                loss_class_train_results, loss_regu_train_results,
                loss_class_val_results, loss_regu_val_results,
                loss_class_test_results, loss_regu_test_results,
                output_directory_models)

    plot_learning_history_all(EPOCH, history, output_directory_models)

    return (history, per_training_duration, log_training_duration)