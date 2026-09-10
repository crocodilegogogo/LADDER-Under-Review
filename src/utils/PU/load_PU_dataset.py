# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from utils.PU.preprocess_raw_data import preprocess_raw_data

def load_PU_raw_data(DATA_DIR, ACT_LABELS, window_size, overlap, INPUT_CHANNEL, SNR, scaler="normalize") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], Dict[str, int]]:
    
    # Call preprocessing function
    # Note: overlap here is actually stride
    X, Y = preprocess_raw_data(DATA_DIR, ACT_LABELS, window_size, overlap, SNR)

    # # Handle label dimensions (N,) -> (N, 1)
    Y = np.expand_dims(Y, 1)

    # Generate label mapping dictionary
    ActID = range(len(ACT_LABELS))
    act2label = dict(zip(ACT_LABELS, ActID))
    label2act = dict(zip(ActID, ACT_LABELS))

    # Adjust data dimensions
    # preprocess returns (N, 1024, 1)
    # Need to swapaxes to (N, 1, 1024) to fit PyTorch Conv1d input convention (usually handled by main.py or inside the model)
    # CWRU load code does swapaxes(1, 2), i.e., from (N, L, C) -> (N, C, L)
    
    X = np.swapaxes(X, 1, 2)

    # CWRU load code does another expand_dims(axis=1) at the end
    # Original code return: np.expand_dims(X_train, axis=1) ... This would become (N, 1, 1, 1024)
    # This might fit some special 2D CNN methods processing 1D signals. Keep consistent with original project:
    
    return np.expand_dims(X, axis=1), Y.squeeze(), label2act, act2label