import math
import os
import numpy as np
from copy import deepcopy
from time import gmtime, strftime

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def add_noise(SNR, raw_data):
    np.random.seed(66)
    random_values = np.random.randn(len(raw_data))  # Gaussian noise
    # cal the power of signal Ps and the power of noise Pn1
    Ps = np.sum(raw_data ** 2) / len(raw_data)
    Pn1 = np.sum(random_values ** 2) / (len(random_values))
    # cal normalization value k
    k = math.sqrt(Ps / (10 ** (SNR / 10) * Pn1))
    # cal the normalized power of noise
    random_values_need = random_values * k
    Pn = np.sum(random_values_need ** 2) / len(random_values_need)
    # cal the signal to noise ratio
    SNR = 10 * math.log10(Ps / Pn)
    # adding noise to the raw_data
    random_values_need = np.expand_dims(random_values_need, axis=1)
    noise_data = random_values_need + raw_data

    return noise_data

label1 = [i for i in range(0,5)]
label2 = [i for i in range(5,10)]
label3 = [i for i in range(10,15)]

def data_load_XJTU(filename,signal_size,SNR,label):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = pd.read_csv(filename)
    fl = fl["Horizontal_vibration_signals"]
    fl = fl.values
    fl = fl.reshape(-1,1)
    if SNR == 'False':
        fl = fl
    else:
        fl = add_noise(SNR, fl)
    data=[]
    lab=[]
    start,end=0,signal_size
    while end<=fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start +=signal_size
        end +=signal_size
    return data, lab

def preprocess_raw_data(read_data_dir, window_size, overlap, SNR,  scaler):
    root = os.path.dirname(read_data_dir)
    WC = os.listdir(root)  # Three working conditions WC0:35Hz12kN WC1:37.5Hz11kN WC2:40Hz10kN
    WC.sort()

    datasetname1 = os.listdir(os.path.join(root, WC[0]))
    datasetname1.sort()
    datasetname2 = os.listdir(os.path.join(root, WC[1]))
    datasetname2.sort()
    datasetname3 = os.listdir(os.path.join(root, WC[2]))
    datasetname3.sort()
    data = []
    lab = []
    for i in range(len(datasetname1)):
        files = os.listdir(os.path.join('/tmp', root, WC[0], datasetname1[i]))
        files.sort(key=lambda x:int(x[:-4]))
        for ii in [-4, -3, -2, -1]:  # Take the data of the last three CSV files
            path1 = os.path.join('/tmp', root, WC[0], datasetname1[i], files[ii])
            data1, lab1 = data_load_XJTU(path1, window_size, SNR, label=label1[i])
            data += data1
            lab += lab1

    for j in range(len(datasetname2)):
        files = os.listdir(os.path.join('/tmp', root, WC[1], datasetname2[j]))
        files.sort(key=lambda x:int(x[:-4]))
        for jj in [-4, -3, -2, -1]:
            path2 = os.path.join('/tmp', root, WC[1], datasetname2[j], files[jj])
            data2, lab2 = data_load_XJTU(path2, window_size, SNR,label=label2[j])
            data += data2
            lab += lab2

    for k in range(len(datasetname3)):
        files = os.listdir(os.path.join('/tmp', root, WC[2], datasetname3[k]))
        files.sort(key=lambda x:int(x[:-4]))
        for kk in [-4, -3, -2, -1]:
            path3 = os.path.join('/tmp', root, WC[2], datasetname3[k], files[kk])
            data3, lab3 = data_load_XJTU(path3, window_size, SNR, label=label3[k])
            data += data3
            lab += lab3

    # preprocessing
    data = np.array(data)
    data = np.squeeze(data)
    data_min = np.min(np.min(data,1),0)
    data_max = np.max(np.max(data,1),0)
    data = (data - data_min)/(data_max-data_min)

    data = np.expand_dims(data, axis=2)

    return data, lab

    # X_train, X_test, Y_train, Y_test = train_test_split(data, lab, test_size=0.2, random_state=0, stratify=lab)

    # X_train = np.expand_dims(X_train, axis=2)
    # X_test = np.expand_dims(X_test, axis=2)

    # return X_train, X_test, Y_train, Y_test
