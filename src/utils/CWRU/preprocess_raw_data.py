import os
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class Preprocess:
    def __init__(self, fs: int = 50) -> None:
        """
        Args:
            fs (int, default=50): Sampling frequency of sensor signals
        """
        self.fs = fs

    def segment_signal(
        self,
        signal: pd.DataFrame,
        window_size: int,
        # overlap_rate: int = 0.5,
        overlap: int,
        res_type: str = "dataframe",
    ) -> List[pd.DataFrame]:
        """Sample sensor signals in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window).
        Args:
            signal (pandas.DataFrame): Raw signal
            window_size (int, default=128): Window size of sliding window to segment raw signals.
            overlap (int, default=10): Overlap of sliding window to segment raw signals.
            res_type (str, default='dataframe'): Type of return value; 'array' or 'dataframe'
        Returns:
            signal_seg (list of pandas.DataFrame): List of segmented sigmal.
        """
        signal_seg = []

        for start_idx in range(0, len(signal) + 1 - window_size, overlap):
            seg = signal.iloc[start_idx : start_idx + window_size].reset_index(drop=True)
            if res_type == "array":
                seg = seg.values
            signal_seg.append(seg)

        if res_type == "array":
            signal_seg = np.array(signal_seg)

        return signal_seg

def preprocess_signal(signal: pd.DataFrame, window_size, overlap) -> pd.DataFrame:
    _signal = signal.copy()
    of = Preprocess()
    # _signal = of.apply_filter(_signal, filter="median")
    # _signal = of.apply_filter(_signal, filter="butterworth")
    _signal = of.segment_signal(_signal, window_size, overlap)
    return _signal


def add_noise(SNR, raw_data):
    # adding noise
    np.random.seed(66)
    random_values = np.random.randn(len(raw_data))  # Gaussian noise
    # cal the power of signal Ps and the power of noise Pn1
    Ps = np.sum(raw_data ** 2) / len(raw_data)
    Pn1 = np.sum(random_values ** 2) / (len(random_values))
    # cal normalization value k
    k = math.sqrt(Ps / (10 ** (SNR / 10) * Pn1))
    random_values_need = random_values * k
    # cal the normalized power of noise
    Pn = np.sum(random_values_need ** 2) / len(random_values_need)
    # cal the signal to noise ratio
    SNR = 10 * math.log10(Ps / Pn)
    # adding noise to the raw_data
    random_values_need = np.expand_dims(random_values_need, axis=1)
    noise_data = random_values_need + raw_data

    return noise_data

def rvs(p, size=1):
    
    rvs = np.array([])
    for i in range(0, size):
        if np.random.rand() <= p:
            a = 1
            rvs = np.append(rvs, a)
        else:
            a = 0
            rvs = np.append(rvs, a)
    return rvs

def impulse_noise(raw_data, pr):
    np.random.seed(33)
    power_w = 1 * (np.sum(raw_data ** 2) / len(raw_data))
    power_z = 0.001*power_w

    random_values = np.random.randn(len(raw_data))
    b = rvs(pr, len(raw_data))
    w = np.sqrt(power_w) * random_values
    z = np.sqrt(power_z) * random_values

    impulse_noise = b * w + z
    impulse_noise = np.expand_dims(impulse_noise, axis=1)
    noise_data = impulse_noise
    return noise_data

labellist_0 = ['0.000-Normal','0.007-Ball', '0.007-InnerRace', '0.007-OuterRace6',
                '0.014-Ball', '0.014-InnerRace', '0.014-OuterRace6',
               '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace6']

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True

def preprocess_raw_data(read_data_dir, window_size, overlap, SNR):

    path      = os.path.dirname(read_data_dir)
    filenames = os.listdir(path)
    filenames.sort()
    DE_times = []
    Label_list = []

    for filename in filenames:

        label = filename[:-4]
        label = label[5:]        # obtain the identified part of file names
        Label_list.append(label)
        filepath = path + '//' + filename
        m = loadmat(filepath)
        keys = list(m.keys())
        # print(keys)
        for key in keys:
            if 'DE_time' in key: # select the data of accelerometer in the drive end
                index1 = key
            if 'FE_time' in key:
                index2 = key
        DE_time_raw = m[index1]

        # adding noise
        if SNR == 'False':
            DE_time_noise = DE_time_raw
        else:
            DE_time_noise = add_noise(SNR, DE_time_raw)

        DE_time = DE_time_noise

        # max-min normalization
        DE_time = (DE_time - np.min(DE_time)) / (np.max(DE_time) - np.min(DE_time))

        DE_times.append(DE_time)

    # choose the first thirty samples as datasets
    for j in range(1,4):
        X = []
        Y = []
        for i in range(j*10,10 + j*10):
            if i == 10:
                i = 9
            if Label_list[i] in labellist_0:
                idx = labellist_0.index(Label_list[i])
            if i == 9:
                i = 10
            # sliding window
            if i == j*10:
                if i == 10:
                    i = 9
                data_window0 = preprocess_signal(pd.DataFrame(DE_times[i]), window_size, window_size)
                data_window = data_window0[:100]
                for k in range(len(data_window)):
                    if k == 0:
                        data0 = np.array(data_window[k].T)
                        data = data0
                    else:
                        data0 =np.array(data_window[k].T)
                        data = np.vstack((data, data0))
                lebal = [idx] * data.shape[0]
                X = data
                Y = lebal
            else:
                data_window0 = preprocess_signal(pd.DataFrame(DE_times[i]), window_size, window_size)
                data_window = data_window0[:100]
                for k in range(len(data_window)):
                    if k == 0:
                        data0 = np.array(data_window[k].T)
                        data = data0
                    else:
                        data0 = np.array(data_window[k].T)
                        data = np.vstack((data, data0))
                lebal = [idx] * data.shape[0]
                X = np.vstack((X, data))
                Y = Y + lebal
        
        if j == 1:
            X_out = X
            Y_out = Y
        else:
            X_out = np.concatenate((X_out, X), axis = 0)
            Y_out = Y_out + Y
        
    X_out = np.expand_dims(X_out, axis=2)
    
    return X_out, Y_out

    #     X_train0, X_test0, Y_train0, Y_test0 = train_test_split(X,Y, test_size=0.3,random_state=0, stratify=Y)

    #     if j == 1:
    #         X_train = X_train0
    #         X_test = X_test0

    #         Y_train = Y_train0
    #         Y_test = Y_test0
    #     else:
    #         X_train = np.vstack((X_train, X_train0))
    #         X_test = np.vstack((X_test, X_test0))

    #         Y_train = Y_train + Y_train0
    #         Y_test = Y_test + Y_test0

    # X_train = np.expand_dims(X_train, axis=2)
    # X_test = np.expand_dims(X_test, axis=2)

    # return X_train, X_test, Y_train, Y_test
