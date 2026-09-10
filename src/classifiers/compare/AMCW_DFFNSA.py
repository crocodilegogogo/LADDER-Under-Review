# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
from utils.utils import *

# # Attempt to import the DWT module from the project; if it fails, use the built-in simple implementation.
# try:
#     from utils.DWT_IDWT.DWT_IDWT_layer import DWT_1D
#     HAS_PROJECT_DWT = True
# except ImportError:
#     HAS_PROJECT_DWT = False

# ==========================================
# Auxiliary Modules: DWT, DCT, Attention
# ==========================================

class Simple_DWT_Layer(nn.Module):
    """
    Alternative solution when DWT_1D from the project cannot be loaded (Conv1d implementation based on Haar wavelet).
    """
    def __init__(self):
        super(Simple_DWT_Layer, self).__init__()
        self.register_buffer('low_filter', torch.tensor([1/math.sqrt(2), 1/math.sqrt(2)]).view(1, 1, 2))
        self.register_buffer('high_filter', torch.tensor([1/math.sqrt(2), -1/math.sqrt(2)]).view(1, 1, 2))

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        low_filter = self.low_filter.repeat(C, 1, 1)
        high_filter = self.high_filter.repeat(C, 1, 1)
        
        # Downsample using stride=2
        low = F.conv1d(x, low_filter, stride=2, groups=C)
        high = F.conv1d(x, high_filter, stride=2, groups=C)
        
        return low, high

class CECAM(nn.Module):
    """
    Cosine-Enhanced Channel Attention Module (CECAM)
    Uses DCT instead of GAP to capture frequency information.
    """
    def __init__(self, in_channels, length, reduction=16):
        super(CECAM, self).__init__()
        self.length = length
        # Generate DCT weight basis (L, 1) - Simplified version, focusing on low frequencies
        k = torch.arange(length, dtype=torch.float32)
        n = torch.arange(length, dtype=torch.float32)
        dct_basis = torch.cos(math.pi / length * (n.unsqueeze(1) + 0.5) * k.unsqueeze(0))
        self.register_buffer('dct_weight', dct_basis[:, 0].view(1, 1, -1)) # Take the first DCT coefficient
        
        reduced_channels = max(in_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # DCT projection replaces Global Average Pooling
        # x: (B, C, L) -> (B, C, 1)
        dct_features = torch.sum(x * self.dct_weight, dim=2, keepdim=True)
        attn = self.fc(dct_features)
        return x * attn + x

class MultiScaleConv(nn.Module):
    """
    Lightweight multi-scale convolution used inside AMCW.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.cross_conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Parallel Depthwise Convs
        self.dw_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=k, padding=k//2, groups=in_channels)
            for k in [1, 3, 5, 7]
        ])
        
        self.fuse_conv = nn.Sequential(
            nn.Conv1d(in_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        y = self.cross_conv(x)
        outs = [conv(y) for conv in self.dw_convs]
        concat = torch.cat(outs, dim=1)
        return self.fuse_conv(concat)

class AMCW(nn.Module):
    """
    Attention-guided Multi-scale Convolution with Wavelet embedding.
    """
    def __init__(self, in_channels, out_channels, length):
        super(AMCW, self).__init__()
        
        # if HAS_PROJECT_DWT:
        #     # Assume using DWT from the project (usually requires specifying wavelet name, e.g., 'db1' or 'bior1.1')
        #     self.dwt = DWT_1D(wavename='db1') 
        #     self.use_simple_dwt = False
        # else:
        #     self.dwt = Simple_DWT_Layer()
        #     self.use_simple_dwt = True
        
        self.dwt = Simple_DWT_Layer()
        
        # After DWT decomposition, the number of channels doubles (Low+High), and the length is halved
        dwt_out_channels = in_channels * 2
        dwt_out_length = length // 2
        
        self.msc = MultiScaleConv(dwt_out_channels, out_channels)
        self.cecam = CECAM(out_channels, dwt_out_length)
        
    def forward(self, x):
        # # x: (B, C, L)
        # if self.use_simple_dwt:
        #     x_l, x_h = self.dwt(x)
        # else:
        #     # DWT_1D in DWT_1D_layer.py returns (L, H)
        #     x_l, x_h = self.dwt(x)
        
        # x: (B, C, L)
        x_l, x_h = self.dwt(x)
        
        # Concatenate low-frequency and high-frequency components -> (B, 2C, L/2)
        x_dwt = torch.cat([x_l, x_h], dim=1)
        
        x_msc = self.msc(x_dwt)
        x_out = self.cecam(x_msc)
        return x_out

class DFFNSA(nn.Module):
    """
    Deep Feature Fusion Network based on Self-Attention.
    """
    def __init__(self, in_channels, d_model, num_heads=4):
        super(DFFNSA, self).__init__()
        self.to_q = nn.Conv1d(in_channels, d_model, 1)
        self.to_k = nn.Conv1d(in_channels, d_model, 1)
        self.to_v = nn.Conv1d(in_channels, d_model, 1)
        
        self.mhsa = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        # Simple position embedding buffer
        self.register_buffer('pos_embed', torch.zeros(1, 2048, d_model)) 
        
        self.local_conv = nn.Sequential(
            nn.Conv1d(d_model * 3, d_model, 1),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )
        self.local_att_conv = nn.Conv1d(d_model, d_model, 1)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, L = x.shape
        Q, K, V = self.to_q(x), self.to_k(x), self.to_v(x)
        
        # --- Global Branch ---
        Q_g = Q.permute(0, 2, 1) + self.pos_embed[:, :L, :]
        K_g = K.permute(0, 2, 1) + self.pos_embed[:, :L, :]
        V_g = V.permute(0, 2, 1)
        
        global_out, _ = self.mhsa(Q_g, K_g, V_g)
        global_out = global_out.permute(0, 2, 1)
        
        # --- Local Branch ---
        x_local = self.local_conv(torch.cat([Q, K, V], dim=1))
        # Simplified version: Use MaxPool and Identity to simulate DCT effect (Local branch in the paper includes DCT and GMP)
        x_gmp = F.adaptive_max_pool1d(x_local, 1)
        local_att = F.softmax(self.local_att_conv(x_gmp * x_local), dim=-1)
        local_out = x_local * local_att
        
        return self.alpha * global_out + self.beta * local_out

# ==========================================
# Main Model Class
# ==========================================

class AMCW_DFFNSA(nn.Module):
    def __init__(self, numclass, data_lenth):
        """
        Parameter description:
        numclass: Number of classes.
        """
        super(AMCW_DFFNSA, self).__init__()
        
        # Assuming input data length is 1024 (common setting), adjust if different
        self.input_len = data_lenth 
        
        # 1. Feature Extraction Block (2 layers of AMCW)
        # Input: (B, 1, 1024) -> DWT -> (2, 512) -> Out 24
        self.amcw1 = AMCW(in_channels=1, out_channels=24, length=self.input_len)
        # Input: (B, 24, 512) -> DWT -> (48, 256) -> Out 48
        self.amcw2 = AMCW(in_channels=24, out_channels=48, length=self.input_len // 2)
        
        # 2. Feature Fusion Block
        self.dffnsa = DFFNSA(in_channels=48, d_model=48, num_heads=4)
        
        self.norm1 = nn.LayerNorm(48)
        self.norm2 = nn.LayerNorm(48)
        self.feed_forward = nn.Sequential(
            nn.Linear(48, 192),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(192, 48)
        )
        self.dropout = nn.Dropout(0.1)
        
        # 3. Output Block
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(48, numclass)

    def forward(self, x):
        # Adapt to WDCNN input format, usually (B, 1, L) or (B, 1, 1, L)
        if x.dim() == 4:
            x = x.squeeze(2) # Change to (B, 1, L)
            
        # Feature Extraction
        f1 = self.amcw1(x)
        f2 = self.amcw2(f1)
        
        # Feature Fusion
        fusion = self.dffnsa(f2)
        
        # Norm & Add (Dimensions need to be adjusted to match LayerNorm)
        fusion_perm = fusion.permute(0, 2, 1) # (B, L, C)
        f2_perm = f2.permute(0, 2, 1)
        
        res1 = self.norm1(fusion_perm + f2_perm)
        ff_out = self.feed_forward(res1)
        res2 = self.norm2(res1 + self.dropout(ff_out))
        
        out_feat = res2.permute(0, 2, 1) # Back to (B, C, L)
        
        # Output
        x_vec = self.avgpool(out_feat).flatten(1)
        logits = self.fc(x_vec)
        
        # Return logits and features (for compatibility with output_bc, regus = network(batch_x) in train_op)
        # regus in WDCNN returns features for regularization; here we also return x_vec
        return logits, x_vec

# ==========================================
# Training and Testing Interface (Directly reuse logic from WDCNN.py)
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