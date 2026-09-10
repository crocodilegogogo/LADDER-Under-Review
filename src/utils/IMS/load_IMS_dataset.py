# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, Tuple
from utils.IMS.preprocess_raw_data import preprocess_raw_data

def load_IMS_raw_data(DATA_DIR, ACT_LABELS, window_size, overlap, INPUT_CHANNEL, SNR) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str], Dict[str, int]]:
    
    # Call core preprocessing function
    # X_train, X_test, Y_train, Y_test = preprocess_raw_data(DATA_DIR, window_size, overlap, SNR)
    X, Y = preprocess_raw_data(DATA_DIR, window_size, overlap, SNR)

    # Label mapping
    ActID = range(len(ACT_LABELS))
    act2label = dict(zip(ACT_LABELS, ActID))
    label2act = dict(zip(ActID, ACT_LABELS))

    # Dimension conversion to fit PyTorch (N, Channel, Length)
    # Preprocessing output: (N, Length, 1) -> Convert to (N, 1, Length)
    # X_train = np.swapaxes(X_train, 1, 2)
    # X_test = np.swapaxes(X_test, 1, 2)
    X = np.swapaxes(X, 1, 2)

    # Add dimension again to fit project interface: (N, 1, 1, Length)
    # Based on load_CWRU_dataset.py logic: return np.expand_dims(X_train, axis=1)...
    # return np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1), Y_train, Y_test, label2act, act2label
    return np.expand_dims(X, axis=1), Y, label2act, act2label

