# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def segment_signal(signal, window_size, stride):
    """Sliding window segmentation"""
    segments = []
    n_samples = len(signal)
    # Ensure slicing as long as length is sufficient, discard the last part if less than one window
    for start in range(0, n_samples - window_size + 1, stride):
        segment = signal[start : start + window_size]
        segments.append(segment)
    return np.array(segments)

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

def process_single_test_set(data_dir, bearing_idx, window_size, stride, SNR, test_set_name=""):
    """
    Process a single Test Set folder.
    Strategy:
    - Normal: First 10% of files
    - Fault:  Last 10% of files
    - Bearing: Specified bearing_idx (uniformly 0 in this case)
    """
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist. Skipping.")
        return [], []

    filenames = sorted(os.listdir(data_dir))
    total_files = len(filenames)
    
    if total_files == 0:
        return [], []

    # Split by 10% ratio
    n_normal_files = int(total_files * 0.10)
    n_fault_files = int(total_files * 0.10)
    
    # Get file list
    normal_files = filenames[:n_normal_files]
    fault_files = filenames[-n_fault_files:]
    
    X_segments = []
    Y_labels = []
    
    print(f"[{test_set_name}] Total: {total_files}, Normal(10%): {n_normal_files}, Fault(10%): {n_fault_files}")

    # --- 1. Process Normal data (Label 0) ---
    for fname in tqdm(normal_files, desc=f"Processing {test_set_name} Normal"):
        filepath = os.path.join(data_dir, fname)
        try:
            # IMS data is usually Tab-separated
            df = pd.read_csv(filepath, sep='\t', header=None)
            signal = df.iloc[:, bearing_idx].values
            
            if SNR is not None and str(SNR) != 'False':
                 signal = add_noise(float(SNR), signal)
            
            # Single file normalization (Min-Max)
            if np.max(signal) != np.min(signal):
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            
            # scaler = StandardScaler()
            # signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
            
            segs = segment_signal(signal, window_size, stride)
            if len(segs) > 0:
                X_segments.append(segs)
                Y_labels.extend([0] * len(segs))
        except Exception as e:
            pass # Ignore files with read errors

    # --- 2. Process Fault data (Label 1) ---
    for fname in tqdm(fault_files, desc=f"Processing {test_set_name} Fault"):
        filepath = os.path.join(data_dir, fname)
        try:
            df = pd.read_csv(filepath, sep='\t', header=None)
            signal = df.iloc[:, bearing_idx].values
            
            if SNR is not None and str(SNR) != 'False':
                 signal = add_noise(float(SNR), signal)
            
            # Single file normalization
            if np.max(signal) != np.min(signal):
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            
            # scaler = StandardScaler()
            # signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
            
            segs = segment_signal(signal, window_size, stride)
            if len(segs) > 0:
                X_segments.append(segs)
                Y_labels.extend([1] * len(segs))
        except Exception as e:
            pass

    return X_segments, Y_labels

def preprocess_raw_data(read_data_dir, window_size, overlap, SNR):
    """
    Read IMS dataset (includes 2nd_test and 3rd_test).
    Args:
        read_data_dir: IMS root directory (containing '2nd_test', '3rd_test' subfolders)
    """
    stride = overlap
    
    # Define sub-dataset names to process
    # Please ensure these two folders exist under datasets/IMS/
    target_datasets = ['2nd_test', '3rd_test']
    # target_datasets = ['2nd_test']
    
    # Force using only Bearing 1 (Index 0)
    target_bearing_idx = 0 
    
    all_X = []
    all_Y = []

    print(f"Starting IMS Preprocessing from root: {read_data_dir}")
    print(f"Target Sub-datasets: {target_datasets}")
    print(f"Using Bearing Index: {target_bearing_idx} (Bearing 1)")
    print(f"Split Strategy: First 10% -> Normal, Last 10% -> Fault")

    for dataset_name in target_datasets:
        sub_dir = os.path.join(read_data_dir, dataset_name)
        
        # Call processing function
        X_sub, Y_sub = process_single_test_set(
            sub_dir, 
            target_bearing_idx, 
            window_size, 
            stride, 
            SNR, 
            test_set_name=dataset_name)
        
        if len(X_sub) > 0:
            all_X.extend(X_sub)
            all_Y.extend(Y_sub)

    if len(all_X) == 0:
        raise ValueError("No data loaded! Please check your dataset paths.")

    # Convert to Numpy array
    # X shape: (Total_Samples, Window_Size)
    X = np.concatenate(all_X, axis=0)
    Y = np.array(all_Y)
    
    # Print final data statistics
    print(f"\nPreprocessing Complete.")
    print(f"Total Samples: {len(Y)}")
    print(f"Normal Samples (Label 0): {np.sum(Y == 0)}")
    print(f"Fault Samples (Label 1): {np.sum(Y == 1)}")

    X = np.expand_dims(X, axis=2)
    
    return X, Y

    # # Split training and testing sets (7:3)
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X, Y, test_size=0.3, random_state=42, stratify=Y
    # )
    
    # # Adjust dimensions to fit AWD-Net: (N, Length) -> (N, Length, 1)
    # X_train = np.expand_dims(X_train, axis=2)
    # X_test = np.expand_dims(X_test, axis=2)
    
    # return X_train, X_test, Y_train, Y_test
