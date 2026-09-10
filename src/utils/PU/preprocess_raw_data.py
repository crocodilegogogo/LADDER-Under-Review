# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import scipy.io
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List

# Define label order for PU dataset, must be consistent with constants.py
labellist_PU = [
    "K001", # Health
    "K002", # Health
    "K003", # Health
    "K004", # Health
    "K005", # Health
    "K006", # Health
    "KA04", 
    "KA15",
    "KA16",
    "KA22",
    "KA30",
    "KB23",
    "KB24",
    "KB27",
    "KI04",
    "KI16",
    "KI17",
    "KI18",
    "KI21",
]

class Preprocess:
    def __init__(self, fs: int = 64000) -> None:
        self.fs = fs

    def segment_signal(self, signal, window_size, overlap, res_type="array"):
        """
        Args:
            signal: (array-like) Raw signal
            window_size: (int) window size
            overlap: (int) This is actually the STRIDE (step size) in this project context
        """
        signal_seg = []
        # Note: The 'overlap' parameter in this project is usually used as stride
        # To implement 50% overlap, the passed overlap parameter should be window_size // 2
        step = overlap 
        
        # Convert DataFrame to numpy for efficiency
        if isinstance(signal, pd.DataFrame):
            signal = signal.values.flatten()
            
        for start_idx in range(0, len(signal) - window_size + 1, step):
            seg = signal[start_idx : start_idx + window_size]
            signal_seg.append(seg)

        if res_type == "array":
            signal_seg = np.array(signal_seg)

        return signal_seg

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
    noise_data = random_values_need + raw_data

    return noise_data.flatten()

# def add_noise(SNR, raw_data):
#     # Reuse existing noise addition logic from the project
#     import math
#     np.random.seed(66)
#     random_values = np.random.randn(len(raw_data))
#     Ps = np.sum(raw_data ** 2) / len(raw_data)
#     Pn1 = np.sum(random_values ** 2) / (len(random_values))
#     k = math.sqrt(Ps / (10 ** (SNR / 10) * Pn1))
#     random_values_need = random_values * k
#     random_values_need = np.expand_dims(random_values_need, axis=1)
    
#     # Ensure dimension matching
#     if raw_data.ndim == 1:
#         raw_data = raw_data[:, np.newaxis]
#     noise_data = random_values_need + raw_data
#     return noise_data.flatten()

def load_mat_data(filepath):
    """Specific loading logic for PU dataset"""
    try:
        mat_data = scipy.io.loadmat(filepath)
        for key in mat_data.keys():
            # PU data usually contains keys starting with bearing codes (e.g., K003)
            # if key.startswith('K'):
            if 'K' in key:
                # Attempt to extract vibration data; adjustment might be needed based on actual structure
                # Assume flattening all data points directly
                data = mat_data[key]
                # Some PU file structures are struct -> Y -> Data
                if data.dtype.names is not None: # If it is a struct
                     if 'Y' in data.dtype.names:
                         y_struct = data['Y'][0][0]
                         if 'Data' in y_struct.dtype.names:
                             return y_struct['Data'][0][0].flatten()
                return data.flatten()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return None

def preprocess_raw_data(read_data_dir, ACT_LABELS, window_size, overlap, SNR):
    """
    Args:
        read_data_dir: Root directory of PU dataset
        window_size: Window size
        overlap: Stride (step size)
        SNR: Signal-to-noise ratio
    """
    # Get all .mat files in the directory
    # Assume filenames contain label names (e.g., N15_M07_F10_K003_1.mat)
    if not os.path.exists(read_data_dir):
        raise FileNotFoundError(f"Directory {read_data_dir} does not exist.")
    
    # X = []
    # Y = []
    
    X = None
    Y = None
    
    for file in ACT_LABELS:
        
        read_data_dir_mat = os.path.join(read_data_dir, file)
        
        filenames = [f for f in os.listdir(read_data_dir_mat) if f.endswith('.mat')]
        filenames.sort()
    
        All_X = []
        All_Y = []
        
        processor = Preprocess()
    
        for filename in filenames:
            # 1. Determine label
            found_label = None
            for label in ACT_LABELS:
                if label in filename:
                    found_label = label
                    break
            
            if found_label is None:
                continue # Skip files not in the list
                
            label_idx = ACT_LABELS.index(found_label)
            # if found_label in ["K001","K002","K003","K004","K005","K006"]:
            #     label_idx = 0
            # else:
            #     label_idx = label_idx - 5
            
            # 2. Load data
            filepath = os.path.join(read_data_dir_mat, filename)
            raw_data = load_mat_data(filepath)
            
            if raw_data is None or len(raw_data) < window_size:
                continue
    
            # 3. Add noise (if needed)
            if SNR is not None and str(SNR) != 'False':
                 raw_data = add_noise(float(SNR), raw_data)
    
            # 4. Data standardization (ANC-Net uses Z-Score)
            # scaler = StandardScaler()
            # raw_data = scaler.fit_transform(raw_data.reshape(-1, 1)).flatten()
            raw_data = (raw_data - np.min(raw_data)) / (np.max(raw_data) - np.min(raw_data))
    
            # 5. Sliding window segmentation
            segments = processor.segment_signal(raw_data, window_size, overlap, res_type="array")
            
            if len(segments) > 0:
                labels = [label_idx] * len(segments)
                All_X.append(segments)
                All_Y.append(labels)
    
            if not All_X:
                raise ValueError("No valid data processed. Check dataset path and filenames.")
    
        All_X = np.concatenate(All_X, axis=0)
        All_Y = np.concatenate(All_Y, axis=0)
        # if X==[] and Y==[]:
        if X is None and Y is None:
            X = All_X
            Y = All_Y
        else:
            X = np.concatenate((X, All_X), axis=0)
            Y = np.concatenate((Y, All_Y), axis=0)

    X = np.expand_dims(X, axis=2)
    
    return X, Y

    # # 6. Split training and testing sets
    # # Random split following project convention
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.3, random_state=0, stratify=Y)

    # # 7. Adjust dimensions to fit model input (N, C, L) 
    # # Add last dimension here, transpose in Load script
    # # CWRU preprocess returns: np.expand_dims(X_train, axis=2) -> (N, L, 1)
    
    # X_train = np.expand_dims(X_train, axis=2)
    # X_test = np.expand_dims(X_test, axis=2)

    # return X_train, X_test, Y_train, Y_test
