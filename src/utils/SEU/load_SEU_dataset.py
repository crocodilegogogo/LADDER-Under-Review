"""Load dataset"""
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.SEU.preprocess_raw_data import preprocess_raw_data  # all_data

def load_SEU_raw_data(DATA_DIR, ACT_LABELS, window_size, overlap,INPUT_CHANNEL,SNR,
                       scaler: str = "normalize",
                       ) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], Dict[str, int]]:

    X, Y = preprocess_raw_data(DATA_DIR,  window_size, overlap,INPUT_CHANNEL, SNR, scaler=scaler)

    X = np.swapaxes(X, 1, 2)
    Y = np.expand_dims(Y, 1)

    ActID = range(len(ACT_LABELS))
    act2label = dict(zip(ACT_LABELS, ActID))
    label2act = dict(zip(ActID, ACT_LABELS))

    return np.expand_dims(X, axis=1), Y.squeeze(), label2act, act2label
                               
    # X_train, X_test, Y_train, Y_test = preprocess_raw_data(DATA_DIR,  window_size, overlap,INPUT_CHANNEL, SNR,
    #                                                        scaler=scaler)

    # y_train = np.expand_dims(Y_train, 1)
    # y_test = np.expand_dims(Y_test, 1)

    # ActID = range(len(ACT_LABELS))
    # act2label = dict(zip(ACT_LABELS, ActID))
    # label2act = dict(zip(ActID, ACT_LABELS))

    # X_train = np.swapaxes(X_train, 1, 2)
    # X_test = np.swapaxes(X_test, 1, 2)

    # return np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1), y_train.squeeze(), y_test.squeeze(), label2act, act2label