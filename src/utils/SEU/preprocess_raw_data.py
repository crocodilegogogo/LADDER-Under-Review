import numpy as np 
from scipy.fftpack import fft
import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import math
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

def impulse_noise(raw_data,pr):
    np.random.seed(33)
    power_w = 2 * (np.sum(raw_data ** 2) / len(raw_data))
    power_z = 0.001*power_w
    random_values = np.random.randn(len(raw_data))
    b = rvs(pr, len(raw_data))
    w = np.sqrt(power_w) * random_values
    z = np.sqrt(power_z) * random_values
    noise_data = b * w + z
    return noise_data

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

    return noise_data

def scale(
    signal: pd.DataFrame, scaler="normalize", minmax_range: Optional[Tuple[int, int]] = (0, 1)
) -> pd.DataFrame:
    if scaler == "normalize":
        signal = StandardScaler().fit_transform(signal)
        return pd.DataFrame(signal, columns=["Channel1"])
    elif scaler == "minmax":
        signal = MinMaxScaler(feature_range=minmax_range).fit_transform(signal)
        return pd.DataFrame(signal, columns=["Channel1"])

def data_pre(read_data_dir, Label_list, i, SNR,
             window_size, overlap, index0, lenth,
             X_train, Y_train):

    for sub_file in Label_list:

        data_path = read_data_dir + '//' + sub_file
        if sub_file == "ball_20_0.csv":
            sub_data = pd.read_csv(data_path, sep=',')
            raw_data = sub_data.iloc[16:, 0:2]  # shape [105 , 8]
            raw_data = np.array(raw_data.apply(pd.to_numeric))
            raw_data = raw_data[:, 1]
        else:
            sub_data = pd.read_csv(data_path, sep='\t')
            raw_data = sub_data.iloc[16:, 1]  # shape [105 , 8]
            raw_data = np.array(raw_data.apply(pd.to_numeric))
        ###
        if SNR == 'False':
            noise_data = raw_data
        else:
            noise_data = add_noise(SNR, raw_data)
        # normalization
        noise_data = scale(pd.DataFrame(noise_data), scaler="minmax")
        noise_data = 2*noise_data - 1
        # use the sliding window to obtain the training dataset
        data = preprocess_signal(pd.DataFrame(noise_data), window_size, overlap)
        data = data[index0:index0+lenth]
        datatrain = np.array(data).squeeze(-1)
        labels = [i] * datatrain.shape[0]
        X = datatrain
        Y = labels

        X_train = np.vstack((X_train, X))
        Y_train = Y_train + Y

        i = 1 + i

    return X_train, Y_train,

def data_pre_init(read_data_dir, Label_list, i, SNR,
             window_size, overlap, index0, lenth,
             X_train, Y_train):

    for sub_file in Label_list:

        data_path = read_data_dir + '//' + sub_file

        sub_data = pd.read_csv(data_path, sep='\t')
        raw_data = sub_data.iloc[16:, 1]  # shape [105 , 8]
        raw_data = np.array(raw_data.apply(pd.to_numeric))

        ###
        if SNR == 'False':
            noise_data = raw_data
        else:
            noise_data = add_noise(SNR, raw_data)
        # normalization
        noise_data = scale(pd.DataFrame(noise_data), scaler="minmax")
        noise_data = 2*noise_data - 1

        # use the sliding window to obtain the training dataset
        data = preprocess_signal(pd.DataFrame(noise_data), window_size, overlap)
        data = data[index0:index0+lenth]
        
        datatrain = np.array(data).squeeze(-1)
        labels  = [i] * datatrain.shape[0]

        X_train = datatrain
        Y_train = labels

    return X_train, Y_train

def preprocess_raw_data(read_data_dir_SEU, window_size, overlap, INPUT_CHANNEL, SNR, scaler):

    X_train = np.array([])
    Y_train = np.array([])
    X_test = np.array([])
    Y_test = np.array([])

    read_data_dir = read_data_dir_SEU+'bearingset'
    Label_list    = [ "health_20_0.csv"]
    i             = 0
    index0        = 200
    lenth         = 150

    # Initialize X_train and Y_train
    X_train, Y_train = data_pre_init(
        read_data_dir, Label_list, i, SNR,
        window_size, overlap, index0, lenth,
        X_train, Y_train
    )
 ###################################################
    read_data_dir = read_data_dir_SEU+'bearingset'
    Label_list = [ "health_30_2.csv"]
    i = 0

    X_train, Y_train= data_pre(
        read_data_dir, Label_list, i, SNR,
        window_size, overlap, index0, lenth,
        X_train, Y_train
    )
#####################################################
    read_data_dir = read_data_dir_SEU+'gearset'
    Label_list = [ "Health_20_0.csv"]
    i = 0

    X_train, Y_train= data_pre(
        read_data_dir, Label_list, i, SNR,
             window_size, overlap, index0, lenth,
             X_train, Y_train,
    )
#####################################################
    read_data_dir = read_data_dir_SEU+'gearset'
    Label_list = ["Health_30_2.csv"]
    i = 0

    X_train, Y_train = data_pre(
        read_data_dir, Label_list, i, SNR,
             window_size, overlap, index0, lenth,
             X_train, Y_train,
    )
######################################################
    read_data_dir = read_data_dir_SEU+'bearingset'
    Label_list = [ "ball_20_0.csv",
                   "comb_20_0.csv",
                   "inner_20_0.csv",
                   "outer_20_0.csv",]
    i = 1

    X_train, Y_train = data_pre(
        read_data_dir, Label_list, i, SNR,
             window_size, overlap, index0, lenth,
             X_train, Y_train
    )
######################################################
    read_data_dir = read_data_dir_SEU+'bearingset'
    Label_list = [ "ball_30_2.csv",
                   "comb_30_2.csv",
                   "inner_30_2.csv",
                   "outer_30_2.csv",]
    i = 1

    X_train, Y_train = data_pre(
        read_data_dir, Label_list, i, SNR,
             window_size, overlap, index0, lenth,
             X_train, Y_train
    )
######################################################
    read_data_dir = read_data_dir_SEU+'gearset'
    Label_list = [ "Chipped_20_0.csv",
                   "Miss_20_0.csv",
                   "Root_20_0.csv",
                   "Surface_20_0.csv", ]
    i = 5

    X_train, Y_train = data_pre(
        read_data_dir, Label_list, i, SNR,
             window_size, overlap, index0, lenth,
             X_train, Y_train
    )

    Label_list = ["Chipped_30_2.csv",
                  "Miss_30_2.csv",
                  "Root_30_2.csv",
                  "Surface_30_2.csv"]
    i = 5

    X_train, Y_train= data_pre(
        read_data_dir, Label_list, i, SNR,
             window_size, overlap, index0, lenth,
             X_train, Y_train
    )
######################################################

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], INPUT_CHANNEL)

    return X_train, Y_train

    # X_train0, X_test0, Y_train0, Y_test0 = train_test_split(X_train,
    #                 Y_train, test_size=0.2, random_state=66, stratify=Y_train)

    # X_train0 = X_train0.reshape(X_train0.shape[0], X_train0.shape[1], INPUT_CHANNEL)
    # X_test0 = X_test0.reshape(X_test0.shape[0], X_test0.shape[1], INPUT_CHANNEL)

    # return X_train0, X_test0, Y_train0, Y_test0
