"""Load dataset"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from utils.MFPT.preprocess_raw_data import preprocess_raw_data

def load_MFPT_raw_data(DATA_DIR, ACT_LABELS, window_size, overlap, INPUT_CHANNEL, SNR,
                       scaler: str = "normalize") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], Dict[str, int]]:
    
    # Call core preprocessing function
    X, Y = preprocess_raw_data(DATA_DIR, window_size, overlap, SNR)

    # Handle label dimensions
    Y = np.expand_dims(Y, 1)

    # Generate label mapping dictionary
    ActID = range(len(ACT_LABELS))
    act2label = dict(zip(ACT_LABELS, ActID))
    label2act = dict(zip(ActID, ACT_LABELS))

    # Dimension conversion (N, Length, 1) -> (N, 1, Length)
    X = np.swapaxes(X, 1, 2)

    # The final output usually needs to add a dimension as Channel (N, 1, 1, Length) or (N, 1, Channel, Length)
    # Refer to the return value of load_CWRU_dataset.py: np.expand_dims(X_train, axis=1)
    return np.expand_dims(X, axis=1), Y.squeeze(), label2act, act2label
