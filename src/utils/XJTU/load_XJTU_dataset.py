"""Load dataset"""
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.XJTU.preprocess_raw_data import preprocess_raw_data

def load_XJTU_raw_data(DATA_DIR, ACT_LABELS, window_size, overlap,INPUT_CHANNEL, SNR,
                       scaler: str = "normalize",
                       ) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[int, str], Dict[str, int]]:
    """Load raw dataset.

        scaler (str): scaler for raw signals, chosen from normalize or minmax
    Returns:
        X_train (pd.DataFrame):
        X_test (pd.DataFrame):
        y_train (pd.DataFrame):
        y_test (pd.DataFrame):
        label2act (Dict[int, str]): Dict of label_id to title_of_class
        act2label (Dict[str, int]): Dict of title_of_class to label_id
    """
    
    X_train, Y_train = preprocess_raw_data(DATA_DIR,  window_size, overlap,SNR, scaler=scaler)
    X_train = np.swapaxes(X_train, 1, 2)
    y_train = np.expand_dims(Y_train, 1)
    
    ActID = range(len(ACT_LABELS))
    act2label = dict(zip(ACT_LABELS, ActID))
    label2act = dict(zip(ActID, ACT_LABELS))
    
    return np.expand_dims(X_train, axis=1), y_train.squeeze(), label2act, act2label
    
    # X_train, X_test, Y_train, Y_test = preprocess_raw_data(DATA_DIR,  window_size, overlap,SNR,
    #                                                        scaler=scaler)


    # y_train = np.expand_dims(Y_train, 1)
    # y_test = np.expand_dims(Y_test, 1)

    # ActID = range(len(ACT_LABELS))
    # act2label = dict(zip(ACT_LABELS, ActID))
    # label2act = dict(zip(ActID, ACT_LABELS))

    # X_train = np.swapaxes(X_train, 1, 2)
    # X_test = np.swapaxes(X_test, 1, 2)


    # return np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1), y_train.squeeze(), y_test.squeeze(), label2act, act2label

