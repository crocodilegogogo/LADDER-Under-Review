import os
import numpy as np
import scipy.io
import math
from sklearn.model_selection import train_test_split

def add_noise(SNR, raw_data):
    if SNR == 'False' or SNR is None:
        return raw_data
        
    np.random.seed(66)
    random_values = np.random.randn(len(raw_data))
    Ps = np.sum(raw_data ** 2) / len(raw_data)
    Pn1 = np.sum(random_values ** 2) / (len(random_values))
    k = math.sqrt(Ps / (10 ** (int(SNR) / 10) * Pn1))
    random_values_need = random_values * k
    # Adjust dimensions to match
    random_values_need = np.expand_dims(random_values_need, axis=1).flatten() 
    
    return raw_data + random_values_need

def load_mat_data(file_path):
    """
    Read the complex mat structure of MFPT.
    """
    try:
        mat = scipy.io.loadmat(file_path)
        if 'bearing' in mat:
            bearing_data = mat['bearing'][0][0]
            signal = bearing_data['gs'].flatten()
            sr = bearing_data['sr'][0][0]
            return signal, sr
        else:
            return None, None
    except:
        return None, None

def preprocess_raw_data(data_dir, window_size, overlap, SNR):
    # Define MFPT file categories
    # Assume your data is stored in the data_dir directory
    files_dict = {
        0: ['baseline_1.mat', 'baseline_2.mat', 'baseline_3.mat'], # Normal
        1: ['OuterRaceFault_1.mat', 'OuterRaceFault_2.mat', 'OuterRaceFault_3.mat'], # Outer Race
        2: ['InnerRaceFault_vload_1.mat', 'InnerRaceFault_vload_2.mat', 'InnerRaceFault_vload_3.mat', 
            'InnerRaceFault_vload_4.mat', 'InnerRaceFault_vload_5.mat', 'InnerRaceFault_vload_6.mat', 
            'InnerRaceFault_vload_7.mat'] # Inner Race
    }

    X_all = []
    Y_all = []
    stride = window_size - overlap # Calculate stride

    print(f"Processing MFPT dataset from {data_dir}...")

    for label, filenames in files_dict.items():
        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"Warning: {filename} not found.")
                continue

            raw_signal, sr = load_mat_data(filepath)
            if raw_signal is None:
                continue

            # 1. Resampling: If ~97k Hz, downsample to ~48k Hz
            if sr > 90000:
                raw_signal = raw_signal[::2]

            # 2. Add noise
            signal = add_noise(SNR, raw_signal)

            # 3. Normalization (Min-Max Normalization to match CWRU style)
            # Note: MFPT usually recommends Z-score, but Min-Max is used here to match the preprocessing style of other datasets in this project.
            if np.max(signal) != np.min(signal):
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

            # 4. Sliding window segmentation
            num_samples = (len(signal) - window_size) // stride + 1
            if num_samples <= 0:
                continue
            
            for i in range(num_samples):
                start = i * stride
                end = start + window_size
                segment = signal[start:end]
                X_all.append(segment)
                Y_all.append(label)

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)

    X_all = np.expand_dims(X_all, axis=2)
    
    return X_all, Y_all

    # # Split training and test sets (maintain consistent stratify with the project)
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     X_all, Y_all, test_size=0.3, random_state=0, stratify=Y_all
    # )

    # # Adjust dimensions to adapt to subsequent swapaxes operations in the project
    # # CWRU code returns (N, Length, 1)
    # X_train = np.expand_dims(X_train, axis=2)
    # X_test = np.expand_dims(X_test, axis=2)

    # return X_train, X_test, Y_train, Y_test