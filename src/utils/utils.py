"""Collection of utility functions"""
from datetime import datetime
from logging import getLogger, Formatter, FileHandler, StreamHandler, DEBUG, WARNING
from decimal import Decimal, ROUND_HALF_UP
from collections import Counter
from utils.CWRU.load_CWRU_dataset import load_CWRU_raw_data
from utils.SEU.load_SEU_dataset import load_SEU_raw_data
from utils.XJTU.load_XJTU_dataset import load_XJTU_raw_data
from utils.PU.load_PU_dataset import load_PU_raw_data
from utils.IMS.load_IMS_dataset import load_IMS_raw_data
from utils.MFPT.load_MFPT_dataset import load_MFPT_raw_data
from utils.download_datasets import ensure_dataset_available
from utils.constants import get_CWRU_dataset_param, get_SEU_dataset_param, get_XJTU_dataset_param, get_PU_dataset_param, get_IMS_dataset_param, \
                                                         get_MFPT_dataset_param, create_classifier
import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as Data
import time
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

# shap.initjs()
logger = getLogger(__name__)

def round_float(f: float, r: float = 0.000001) -> float:
    return float(Decimal(str(f)).quantize(Decimal(str(r)), rounding=ROUND_HALF_UP))


def round_list(l: List[float], r: float = 0.000001) -> List[float]:
    return [round_float(f, r) for f in l]


def round_dict(d: Dict[Any, Any], r: float = 0.000001) -> Dict[Any, Any]:
    return {key: round(d[key], r) for key in d.keys()}

def round(arg: Any, r: float = 0.000001) -> Any:
    if type(arg) == float or type(arg) == np.float64 or type(arg) == np.float32:
        return round_float(arg, r)
    elif type(arg) == list or type(arg) == np.ndarray:
        return round_list(arg, r)
    elif type(arg) == dict:
        return round_dict(arg, r)
    else:
        logger.error(f"Arg type {type(arg)} is not supported")
        return arg

def load_raw_data(dataset_name,CUR_DIR,i):

    # Download only the selected dataset when its expected directory is absent.
    # Set LADDER_AUTO_DOWNLOAD=0 to require manual dataset placement instead.
    ensure_dataset_available(dataset_name, CUR_DIR)

    if dataset_name == 'CWRU_10':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID,\
          WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes = get_CWRU_dataset_param(
            CUR_DIR, dataset_name)
        # SNR = 'False'
        SNR = -10 + 2 * i # SNRs range from -10 to 10
        X, Y, label2act, act2label = load_CWRU_raw_data(DATA_DIR,  ACT_LABELS,
                                                        WINDOW_SIZE, OVERLAP,INPUT_CHANNEL, SNR)

    if dataset_name == 'SEU':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID,\
          WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes = get_SEU_dataset_param(
            CUR_DIR, dataset_name)
        # SNR = 'False'
        SNR = -10 + 2 * i
        X, Y, label2act, act2label = load_SEU_raw_data(DATA_DIR, ACT_LABELS,
                                                       WINDOW_SIZE, OVERLAP,INPUT_CHANNEL, SNR)

    if dataset_name == 'XJTU':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID,\
          WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes = get_XJTU_dataset_param(
            CUR_DIR, dataset_name)
        # SNR = 'False'
        SNR = -10 + 2 * i
        X, Y, label2act, act2label = load_XJTU_raw_data(DATA_DIR, ACT_LABELS,
                                                        WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR)

    if dataset_name == 'PU':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID,\
            WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes = get_PU_dataset_param(
                CUR_DIR, dataset_name)
        # SNR = 'False'
        SNR = -10 + 2 * i
        X, Y, label2act, act2label = load_PU_raw_data(DATA_DIR, ACT_LABELS,
                                                      WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR)

    if dataset_name == 'IMS':
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID,\
          WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes = get_IMS_dataset_param(
            CUR_DIR, dataset_name)
        # SNR = 'False'
        SNR = -10 + 2 * i
        
        X, Y, label2act, act2label = load_IMS_raw_data(DATA_DIR, ACT_LABELS,
                                                       WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR)

    elif dataset_name == 'MFPT':
        # Get parameters
        DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID,\
            WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes = get_MFPT_dataset_param(
                CUR_DIR, dataset_name)
        # SNR = 'False'
        SNR = -10 + 2 * i
        X, Y, label2act, act2label = load_MFPT_raw_data(DATA_DIR, ACT_LABELS,
                                                        WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR)

    return X, Y, label2act, act2label, ACT_LABELS, ActID, MODELS_COMP_LOG_DIR, INPUT_CHANNEL, \
           SNR, nb_classes

def create_cuda_classifier(dataset_name, classifier_name, INPUT_CHANNEL, data_length, nb_classes):
    classifier, classifier_func = create_classifier(dataset_name, classifier_name, INPUT_CHANNEL,
                                                    data_length, nb_classes)
    classifier.cuda()
    print(classifier)
    classifier_parameter = get_parameter_number(classifier)

    return classifier, classifier_func, classifier_parameter

def shuffle_trainset(X_train, y_train):
    indices = np.arange(X_train.shape[0])
    np.random.seed(77)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    return X_train, y_train

def shuffle_train_test(X_train, y_train, X_test, y_test):
    x_dataset = np.concatenate((X_train, X_test), axis=0)
    y_dataset = np.concatenate((y_train, y_test), axis=0)
    indices = np.arange(x_dataset.shape[0])
    np.random.seed(66)
    np.random.shuffle(indices)
    x_dataset = x_dataset[indices]
    bb = x_dataset.squeeze()
    y_dataset = y_dataset[indices]
    X_train = x_dataset[:len(y_train), :, :, :]
    y_train = y_dataset[:len(y_train)]
    X_test = x_dataset[len(y_train):, :, :, :]
    y_test = y_dataset[len(y_train):]

    return X_train, y_train, X_test, y_test

def log_dataset_info_training_parm(X, Y, ACT_LABELS, ActID, label2act, \
                                   nb_classes, BATCH_SIZE, EPOCH, LR, CV_SPLITS,\
                                   SNR):
    logger.debug(f"X_shape = {X.shape}")
    logger.debug(f"Y_shape = {Y.shape}")
    # logger.debug(f"Cal_Attitude_Angle = {cal_attitude_angle}")
    logger.debug(f"ACT_LABELS = {ACT_LABELS}")
    logger.debug(f"ActID = {ActID}")

    # check the category imbalance
    # check_class_balance(y_train.flatten(), y_test.flatten(), label2act=label2act, n_class=nb_classes)

    # log the hyper-parameters
    logger.debug(f"BATCH_SIZE : {BATCH_SIZE}, EPOCH : {EPOCH}, LR : {LR}, CV_SPLITS : {CV_SPLITS}, SNR : {SNR}" )

def check_class_balance(
        y_train: np.ndarray, y_test: np.ndarray, label2act: Dict[int, str], n_class: int = 12
) -> None:
    c_train = Counter(y_train)
    c_test = Counter(y_test)

    for c, mode in zip([c_train, c_test], ["train", "test"]):
        logger.debug(f"{mode} labels")
        len_y = sum(c.values())
        for label_id in range(n_class):
            logger.debug(
                f"{label2act[label_id]} ({label_id}): {c[label_id]} samples ({c[label_id] / len_y * 100:.04} %)"
            )

def trn_test_data_cv_split(X_train, y_train, train_index, test_index):
    # data and classifier preparation
    X_tr = X_train[train_index, :]
    X_test = X_train[test_index, :]
    Y_tr = y_train[train_index]
    Y_test = y_train[test_index]
    return X_tr, X_test, Y_tr, Y_test

def find_power(N):
    t = 0
    while 2**t < N:
        t += 1
    return t

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path

def logging_settings(classifier_name, CUR_DIR, dataset_name, SNR):
    # Logging settings
    # "-" + str(SNR) +
    EXEC_TIME = classifier_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    LOG_DIR = os.path.join(CUR_DIR, f"logs", dataset_name, classifier_name, f"{EXEC_TIME}")
    MODEL_DIR = os.path.join(CUR_DIR, f"saved_model", dataset_name, classifier_name)
    create_directory(LOG_DIR)  # Create log directory

    # create log object with classifier_name
    cur_classifier_log = getLogger(classifier_name)
    # set recording format
    formatter = Formatter("%(levelname)s: %(asctime)s: %(filename)s: %(funcName)s: %(message)s")
    # create FileHandler with current LOG_DIR and format
    fileHandler = FileHandler(f"{LOG_DIR}/{EXEC_TIME}.log")
    fileHandler.setFormatter(formatter)
    streamHandler = StreamHandler()
    streamHandler.setFormatter(formatter)

    mpl_logger = getLogger("matplotlib")  # Suppress matplotlib logging
    mpl_logger.setLevel(WARNING)

    cur_classifier_log.setLevel(DEBUG)
    cur_classifier_log.addHandler(fileHandler)
    cur_classifier_log.addHandler(streamHandler)

    # important! get current logger with its name (the name is set with classifier_name)
    logger = getLogger(classifier_name)
    logger.setLevel(DEBUG)
    logger.debug(f"{LOG_DIR}/{EXEC_TIME}.log")

    return EXEC_TIME, LOG_DIR, MODEL_DIR, logger

def initialize_saving_variables():
    test_preds = []
    models = []
    scores: Dict[str, Dict[str, List[Any]]] = {
        "logloss": {"train": [], "valid": [], "test": []},
        "accuracy": {"train": [], "valid": [], "test": []},
        "macro-precision": {"train": [], "valid": [], "test": []},
        "macro-recall": {"train": [], "valid": [], "test": []},
        "macro-f1": {"train": [], "valid": [], "test": []},
        "weighted-f1": {"train": [], "valid": [], "test": []},
        "micro-f1": {"train": [], "valid": [], "test": []},
        "per_class_f1": {"train": [], "valid": [], "test": []},
        "confusion_matrix": {"train": [], "valid": [], "test": []},
    }
    log_training_duration = []

    return test_preds, models, scores, log_training_duration

def log_trn_val_test_dataset_net_info(fold_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                      y_test, nb_classes, classifier_parameter, classifier):
    # log train, validation and test data, log the network
    if fold_id == 0:
        logger.debug(
            f"X_train_shape={X_tr.shape}, X_validation_shape={X_val.shape}, X_test_shape={X_test.shape}")
        logger.debug(
            f"Y_train_shape={Y_tr.shape}, Y_validation_shape={Y_val.shape}, y_test_shape={y_test.shape}")
        logger.debug(f"num of categories = {nb_classes}")
        logger.debug(f"num of network parameter = {classifier_parameter}")
        logger.debug(f"the architecture of the network = {classifier}")

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Network_Total_Parameters:', total_num, 'Network_Trainable_Parameters:', trainable_num)
    return {'Total': total_num, 'Trainable': trainable_num}

def get_test_loss_acc(net, loss_function, x_data, y_data, test_split=1):
    loss_sum_data = torch.tensor(0)
    true_sum_data = torch.tensor(0)
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset=torch_dataset,
                                  batch_size=x_data.shape[0] // test_split,
                                  shuffle=False)
    for step, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            output_bc = net(x)[0]
            # out = net(x)[1] ##########
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()
            loss_bc = loss_function(output_bc, y)
            #+ loss_function1(out, x) ############
            true_num_bc = torch.sum(pred_bc == y).data
            loss_sum_data = loss_sum_data + loss_bc
            true_sum_data = true_sum_data + true_num_bc

    loss = loss_sum_data.data.item() / y_data.shape[0]
    acc = true_sum_data.data.item() / y_data.shape[0]
    return loss, acc

def get_test_loss_acc_lifting(net, loss_function, x_data, y_data, test_split=1):
    loss_sum_regu = torch.tensor(0)
    loss_sum_class = torch.tensor(0)
    loss_sum_data = torch.tensor(0)
    true_sum_data = torch.tensor(0)
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset=torch_dataset,
                                  batch_size=x_data.shape[0] // test_split,
                                  shuffle=False)
    for step, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            output_bc, regus = net(x)
            # get loss
            loss_class = loss_function(output_bc, y)
            loss_total = loss_class
            # If no regularisation used, None inside regus
            if regus[0]:
                loss_regu = sum(regus)
                loss_total += loss_regu

            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()
            true_num_bc = torch.sum(pred_bc == y).data
            loss_sum_data = loss_sum_data + loss_total
            loss_sum_regu = loss_sum_regu + loss_regu
            loss_sum_class = loss_sum_class + loss_class
            true_sum_data = true_sum_data + true_num_bc

    loss = loss_sum_data.data.item() / y_data.shape[0]
    acc = true_sum_data.data.item() / y_data.shape[0]
    loss_class = loss_sum_class.data.item() / y_data.shape[0]
    loss_regu = loss_sum_regu.data.item() / y_data.shape[0]
    return loss, acc, loss_class, loss_regu

def get_test_loss_acc_lifting_gumbel(net, loss_function, x_data, y_data,test_flag=False, test_split=1):
    loss_sum_regu = torch.tensor(0)
    loss_sum_class = torch.tensor(0)
    loss_sum_data = torch.tensor(0)
    true_sum_data = torch.tensor(0)
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset=torch_dataset,
                                  batch_size=x_data.shape[0] // test_split,
                                  shuffle=False)
    for step, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            output_bc, regus = net(x, test_flag)
            # get loss
            loss_class = loss_function(output_bc, y)
            loss_total = loss_class
            # If no regularisation used, None inside regus
            if regus[0]:
                loss_regu = sum(regus)
                loss_total += loss_regu

            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            pred_bc = torch.max(output_bc, 1)[1].data.cuda().squeeze()
            true_num_bc = torch.sum(pred_bc == y).data
            loss_sum_data = loss_sum_data + loss_total
            loss_sum_regu = loss_sum_regu + loss_regu
            loss_sum_class = loss_sum_class + loss_class
            true_sum_data = true_sum_data + true_num_bc

    loss = loss_sum_data.data.item() / y_data.shape[0]
    acc = true_sum_data.data.item() / y_data.shape[0]
    loss_class = loss_sum_class.data.item() / y_data.shape[0]
    loss_regu = loss_sum_regu.data.item() / y_data.shape[0]
    return loss, acc, loss_class, loss_regu

def save_models(net, output_directory_models,
                loss_train, loss_train_results,
                accuracy_validation, accuracy_validation_results):
    output_directory_best_val = output_directory_models + 'best_validation_model.pkl'
    if accuracy_validation >= max(accuracy_validation_results):
        torch.save(net.state_dict(), output_directory_best_val)
    if loss_train <= min(loss_train_results):
        torch.save(net.state_dict(), output_directory_best_val)

def log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results,
                loss_validation_results, accuracy_validation_results, output_directory_models):
    history = pd.DataFrame(data=np.zeros((EPOCH, 5), dtype=np.float64),
                           columns=['train_acc', 'train_loss', 'val_acc', 'val_loss', 'lr'])
    history['train_acc'] = accuracy_train_results
    history['train_loss'] = loss_train_results
    history['val_acc'] = accuracy_validation_results
    history['val_loss'] = loss_validation_results
    history['lr'] = lr_results

    # load saved models, predict, cal metrics and save logs
    history.to_csv(output_directory_models + 'history.csv', index=False)

    return history

def log_history_all(EPOCH, lr_results, loss_train_results, accuracy_train_results,
                    loss_validation_results, accuracy_validation_results,
                    loss_test_results, accuracy_test_results,
                    loss_class_train_results, loss_regu_train_results,
                    loss_class_val_results, loss_regu_val_results,
                    loss_class_test_results, loss_regu_test_results,
                    output_directory_models):

    history = pd.DataFrame(data=np.zeros((EPOCH, 9), dtype=np.float64),
                           columns=['train_acc', 'train_loss', 'val_acc', 'val_loss',
                                    'train_class_loss', 'val_class_loss',
                                    'train_regu_loss', 'val_regu_loss',
                                    'lr'])
    history['train_acc'] = accuracy_train_results
    history['train_loss'] = loss_train_results
    history['train_class_loss'] = loss_class_train_results
    history['train_regu_loss'] = loss_regu_train_results
    history['val_acc'] = accuracy_validation_results
    history['val_loss'] = loss_validation_results
    history['val_class_loss'] = loss_class_val_results
    history['val_regu_loss'] = loss_regu_val_results

    history['lr'] = lr_results

    # load saved models, predict, cal metrics and save logs
    history.to_csv(output_directory_models + 'history.csv', index=False)

    return history

def plot_learning_history(EPOCH, history, path):
    """Plot learning curve
    Args:
        fit (Any): History object
        path (str, default="history.png")
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    axL.plot(history["train_loss"], label="train")
    axL.plot(history["val_loss"], label="validation")
    axL.set_title("Loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")

    axR.plot(history["train_acc"], label="train")
    axR.plot(history["val_acc"], label="validation")
    axR.set_title("Accuracy")
    axR.set_xlabel("epoch")
    axR.set_ylabel("accuracy")
    axR.legend(loc="upper right")

    fig.savefig(path + 'history.png')
    plt.close()

def plot_learning_history_all(EPOCH, history, path):
    """Plot learning curve
    Args:
        fit (Any): History object
        path (str, default="history.png")
    """
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    axL.plot(history["train_loss"], label="train")
    axL.plot(history["train_class_loss"], label="train_class")
    axL.plot(history["train_regu_loss"], label="train_regu")
    axL.plot(history["val_loss"], label="validation")
    axL.plot(history["val_class_loss"], label="val_class")
    axL.plot(history["val_regu_loss"], label="val_regu")
    axL.set_title("Loss")
    axL.set_xlabel("epoch")
    axL.set_ylabel("loss")
    axL.legend(loc="upper right")

    axR.plot(history["train_acc"], label="train")
    axR.plot(history["val_acc"], label="validation")
    axR.set_title("Accuracy")
    axR.set_xlabel("epoch")
    axR.set_ylabel("accuracy")
    axR.legend(loc="upper right")

    fig.savefig(path + 'history.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["val_class_loss"], label="val_class")
    plt.plot(history["val_regu_loss"], label="val_regu")
    plt.plot(history["train_class_loss"], label="train_class")
    plt.plot(history["train_regu_loss"], label="train_regu")
    plt.plot(history["train_class_loss"], label="train_class")
    plt.plot(history["train_regu_loss"], label="train_regu")
    plt.legend("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.savefig(path + 'history_loss.png')
    plt.close()

def model_predict_gumbel(net, x_data, y_data, test_flag,test_split=1):
    predict = []
    output = []
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    # torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data))
    data_loader = Data.DataLoader(dataset=torch_dataset,
                                  batch_size=x_data.shape[0] // test_split,
                                  shuffle=False)

    for step, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # y = y.cuda()
            output_bc = net(x, test_flag)[0]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            out = output_bc.cpu().data.numpy()
            output.extend(out)
    return output

def model_predict(net, x_data, y_data, test_split=1):
    predict = []
    output = []
    torch_dataset = Data.TensorDataset(torch.FloatTensor(x_data), torch.tensor(y_data).long())
    data_loader = Data.DataLoader(dataset=torch_dataset,
                                  batch_size=x_data.shape[0] // test_split,
                                  shuffle=False)

    for step, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # y = y.cuda()
            output_bc = net(x)[0]
            if len(output_bc.shape) == 1:
                output_bc.unsqueeze_(dim=0)
            out = output_bc.cpu().data.numpy()
            output.extend(out)
    return output

def predict_tr_val_test(network, nb_classes, LABELS,
                        train_x, val_x, test_x,
                        train_y, val_y, test_y,
                        scores, per_training_duration,
                        fold_id, output_directory_models,
                        test_split):
    # generate network objects
    network_obj = network
    # load best saved validation models
    best_validation_model = output_directory_models + 'best_validation_model.pkl'
    network_obj.load_state_dict(torch.load(best_validation_model))
    network_obj.eval()
    # get outputs of best saved validation models by concat them, input: train_x, val_x, test_x
    pred_train = np.array(model_predict(network_obj, train_x, train_y, test_split))
    pred_valid = np.array(model_predict(network_obj, val_x, val_y, test_split))
    start_time = time.time()
    pred_test  = np.array(model_predict(network_obj, test_x, test_y, test_split))
    inference_time = (time.time() - start_time) / test_x.shape[0]

    # record the metrics of each cross validation time, initialize the score per CV
    score: Dict[str, Dict[str, List[Any]]] = {
        "logloss": {"train": [], "valid": [], "test": []},
        "accuracy": {"train": [], "valid": [], "test": []},
        "macro-precision": {"train": [], "valid": [], "test": []},
        "macro-recall": {"train": [], "valid": [], "test": []},
        "macro-f1": {"train": [], "valid": [], "test": []},
        "weighted-f1": {"train": [], "valid": [], "test": []},
        "micro-f1": {"train": [], "valid": [], "test": []},
        "per_class_f1": {"train": [], "valid": [], "test": []},
        "confusion_matrix": {"train": [], "valid": [], "test": []},
    }

    loss_function = nn.CrossEntropyLoss()

    for pred, X, y, mode in zip(
            [pred_train, pred_valid, pred_test], [train_x, val_x, test_x], [train_y, val_y, test_y],
            ["train", "valid", "test"]
    ):
        loss, acc = get_test_loss_acc(network_obj, loss_function, X, y, test_split)
        pred = pred.argmax(axis=1)
        # y is already the argmaxed category
        scores["logloss"][mode].append(loss)
        scores["accuracy"][mode].append(acc)
        scores["macro-precision"][mode].append(precision_score(y, pred, average="macro"))
        scores["macro-recall"][mode].append(recall_score(y, pred, average="macro"))
        scores["macro-f1"][mode].append(f1_score(y, pred, average="macro"))
        scores["weighted-f1"][mode].append(f1_score(y, pred, average="weighted"))
        scores["micro-f1"][mode].append(f1_score(y, pred, average="micro"))
        scores["per_class_f1"][mode].append(f1_score(y, pred, average=None))
        scores["confusion_matrix"][mode].append(confusion_matrix(y, pred))

        # record the metrics of each cross validation time
        score["logloss"][mode].append(loss)
        score["accuracy"][mode].append(acc)
        score["macro-precision"][mode].append(precision_score(y, pred, average="macro"))
        score["macro-recall"][mode].append(recall_score(y, pred, average="macro"))
        score["macro-f1"][mode].append(f1_score(y, pred, average="macro"))
        score["weighted-f1"][mode].append(f1_score(y, pred, average="weighted"))
        score["micro-f1"][mode].append(f1_score(y, pred, average="micro"))
        score["per_class_f1"][mode].append(f1_score(y, pred, average=None))
        score["confusion_matrix"][mode].append(confusion_matrix(y, pred))

    save_metrics_per_cv(score, per_training_duration,
                        fold_id, nb_classes, LABELS,
                        output_directory_models)

    return pred_train, pred_valid, pred_test, scores, inference_time

def save_metrics_per_cv(score, per_training_duration,
                        fold_id, nb_classes, LABELS,
                        output_directory_models):

    # save training time
    per_training_duration_pd = pd.DataFrame(data=per_training_duration,
                                            index=["training duration"],
                                            columns=["Cross_Validation_Fold_" + str(fold_id)])
    per_training_duration_pd.to_csv(output_directory_models + 'score.csv', index=True)

    # save "logloss", "accuracy", "precision", "recall", "f1"
    score_pd = pd.DataFrame(data=np.zeros((7, 3), dtype=np.float64),
                            index=["logloss", "accuracy", "macro-precision", "macro-recall",
                                   "macro-f1", "weighted-f1", "micro-f1"],
                            columns=["train", "valid", "test"])

    for row in score_pd.index:
        for column in score_pd.columns:
            score_pd.loc[row, column] = score[row][column][0]
    score_pd.to_csv(output_directory_models + 'score.csv', index=True, mode='a+')

    # save "per_class_f1"
    pd.DataFrame(["per_class_f1"]).to_csv(output_directory_models + 'score.csv', index=False, header=False, mode='a+')
    per_class_f1_pd = pd.DataFrame(data=np.zeros((3, nb_classes), dtype=np.float64),
                                   index=["train", "valid", "test"], columns=LABELS)

    for row in per_class_f1_pd.index:
        for (i, column) in enumerate(per_class_f1_pd.columns):
            per_class_f1_pd.loc[row, column] = score["per_class_f1"][row][0][i]
    per_class_f1_pd.to_csv(output_directory_models + 'score.csv', index=True, mode='a+')

    # save confusion_matrix
    for key in score['confusion_matrix'].keys():
        pd.DataFrame(["confusion_matrix_" + key]).to_csv(output_directory_models + 'score.csv', index=False,
                                                         header=False, mode='a+')
        each_confusion_matrix = pd.DataFrame(data=np.zeros((nb_classes, nb_classes), dtype=np.float64),
                                             index=LABELS, columns=LABELS)
        for (i, row) in enumerate(each_confusion_matrix.index):
            for (j, column) in enumerate(each_confusion_matrix.columns):
                each_confusion_matrix.loc[row, column] = score['confusion_matrix'][key][0][i][j]
        each_confusion_matrix.to_csv(output_directory_models + 'score.csv', index=True, mode='a+')

def save_metrics_all(score, per_training_duration,
                        nb_classes, LABELS,
                        output_directory_models):
    # save training time
    per_training_duration_pd = pd.DataFrame(data=per_training_duration,
                                            index=["training duration"],
                                            columns=["Train_Validation_shuffle"])
    per_training_duration_pd.to_csv(output_directory_models + 'score.csv', index=True)

    # save "logloss", "accuracy", "precision", "recall", "f1"
    score_pd = pd.DataFrame(data=np.zeros((7, 2), dtype=np.float64),
                            index=["logloss", "accuracy", "macro-precision", "macro-recall",
                                   "macro-f1", "weighted-f1", "micro-f1"],
                            columns=["train", "test"])
    for row in score_pd.index:
        for column in score_pd.columns:
            score_pd.loc[row, column] = score[row][column][0]
    score_pd.to_csv(output_directory_models + 'score.csv', index=True, mode='a+')

    # save "per_class_f1"
    pd.DataFrame(["per_class_f1"]).to_csv(output_directory_models + 'score.csv', index=False, header=False, mode='a+')
    per_class_f1_pd = pd.DataFrame(data=np.zeros((2, nb_classes), dtype=np.float64),
                                   index=["train", "test"], columns=LABELS)
    for row in per_class_f1_pd.index:
        for (i, column) in enumerate(per_class_f1_pd.columns):
            per_class_f1_pd.loc[row, column] = score["per_class_f1"][row][0][i]
    per_class_f1_pd.to_csv(output_directory_models + 'score.csv', index=True, mode='a+')

    # save confusion_matrix
    for key in score['confusion_matrix'].keys():
        pd.DataFrame(["confusion_matrix_" + key]).to_csv(output_directory_models + 'score.csv', index=False,
                                                         header=False, mode='a+')
        each_confusion_matrix = pd.DataFrame(data=np.zeros((nb_classes, nb_classes), dtype=np.float64),
                                             index=LABELS, columns=LABELS)
        for (i, row) in enumerate(each_confusion_matrix.index):
            for (j, column) in enumerate(each_confusion_matrix.columns):
                each_confusion_matrix.loc[row, column] = score['confusion_matrix'][key][0][i][j]
        each_confusion_matrix.to_csv(output_directory_models + 'score.csv', index=True, mode='a+')

def log_every_CV_score(logger, log_training_duration, scores, label2act, nb_classes,SNR, CV_SPLITS=1):
    for i in range(CV_SPLITS):
        # Log Every Cross Validation Scores
        logger.debug("---Per Cross Validation Scores, Fold" + str(i) + "---" + str(SNR))

        # log per CV training time
        logger.debug(f"Training Duration = {log_training_duration[i]}s")

        for mode in ["train", "test"]:
            # log the average of "logloss", "accuracy", "precision", "recall", "f1"
            logger.debug(f"---{mode}---")
            logger.debug(
                f"logloss={round(scores['logloss'][mode][i])}, accuracy={round(scores['accuracy'][mode][i])},\
                    macro-precision={round(scores['macro-precision'][mode][i])}, macro-recall={round(scores['macro-recall'][mode][i])},\
                        macro-f1={round(scores['macro-f1'][mode][i])}, weighted-f1={round(scores['weighted-f1'][mode][i])},\
                            micro-f1={round(scores['micro-f1'][mode][i])}")

            # log the average of "per_class_f1"
            class_f1_mat = scores["per_class_f1"][mode]
            class_f1_result = {}
            for class_id in range(nb_classes):
                per_class_f1 = class_f1_mat[i][class_id]
                class_f1_result[label2act[class_id]] = per_class_f1
            logger.debug(f"per-class f1={round(class_f1_result)}")

def log_averaged_CV_scores(logger, log_training_duration, scores, label2act, nb_classes,SNR, CV_SPLITS=1):
    # Log Cross Validation Scores by averaging them
    logger.debug("---Cross Validation Averaged Scores---"+ str(SNR))
    # log the average of training time
    logger.debug(f"Averaged Training Duration = {(np.mean(log_training_duration))}s")

    for mode in ["train", "test"]:

        # log the average of "logloss", "accuracy", "precision", "recall", "f1"
        logger.debug(f"---{mode}---")
        for metric in ["logloss", "accuracy", "macro-precision", "macro-recall", "macro-f1", "weighted-f1", "micro-f1", "acc_std"]:
            if metric != "acc_std":
                logger.debug(f"{metric}={round(np.mean(scores[metric][mode]))}")
            else:
                metric_flag = "accuracy"
                logger.debug(f"{metric}={round(np.var(scores[metric_flag][mode]))}")

        # log the average of "per_class_f1"
        class_f1_mat = scores["per_class_f1"][mode]
        class_f1_result = {}
        for class_id in range(nb_classes):
            mean_class_f1 = np.mean([class_f1_mat[i][class_id] for i in range(CV_SPLITS)])
            class_f1_result[label2act[class_id]] = mean_class_f1
        logger.debug(f"per-class f1={round(class_f1_result)}")

def log_ensembled_CV_scores(logger, y_test, test_preds, SNR):
    # Output the ensemble of five CV folds by averaging the predicts
    logger.debug("---Final Test Scores Ensemble-Averaged over Folds---" + str(SNR))
    # test_pred = y_test
    test_pred = np.mean(test_preds, axis=0).argmax(axis=1)  # average over folds
    logger.debug(f"accuracy={accuracy_score(y_test, test_pred)}")
    logger.debug(f"macro-precision={precision_score(y_test, test_pred, average='macro')}")
    logger.debug(f"macro-recall={recall_score(y_test, test_pred, average='macro')}")
    logger.debug(f"macro-f1={f1_score(y_test, test_pred, average='macro')}")
    logger.debug(f"weighted-f1={f1_score(y_test, test_pred, average='weighted')}")
    logger.debug(f"micro-f1={f1_score(y_test, test_pred, average='micro')}")
    logger.debug(f"per-class f1={f1_score(y_test, test_pred, average=None)}")
    logger.debug(f"confusion_matrix={confusion_matrix(y_test, test_pred)}")

    return test_pred

def plot_confusion_matrix(
        cms: Dict[str, np.ndarray],
        labels: Optional[List[str]] = None,
        path: str = "confusion_matrix.png",
) -> None:
    """Plot confusion matrix"""
    # Cal the ensembled confusion_matrix by averaging them
    cms = [np.mean(cms[mode], axis=0) for mode in ["train", "test"]]

    fig, ax = plt.subplots(ncols=3, figsize=(20, 7))
    for i, (cm, mode) in enumerate(zip(cms, ["train",  "test"])):
        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            square=True,
            vmin=0,
            vmax=1.0,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax[i],
        )
        ax[i].set_xlabel("Predicted label")
        ax[i].set_ylabel("True label")
        ax[i].set_title(f"Averaged confusion matrix - {mode}")

    plt.tight_layout()
    fig.savefig(path)
    plt.close()


def save_classifiers_comparison(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name, scores, y_test, SNR, log_training_duration, inference_time):
    for i in range(len(CLASSIFIERS)):
        if i == 0:
            CLASSIFIERS_names = CLASSIFIERS[0] + '&'
        else:
            CLASSIFIERS_names = CLASSIFIERS_names + CLASSIFIERS[i] + '&'
    classifiers_comparison_log_dir = MODELS_COMP_LOG_DIR + CLASSIFIERS_names + str(SNR) + '-comparison' + '.csv'

    # record Averaged_CV_scores
    averaged_score_pd = pd.DataFrame(data=np.zeros((9, 1), dtype=np.float64),
                                     index=["accuracy", "macro-precision", "macro-recall", "macro-f1",
                                            "weighted-f1", "micro-f1", "acc_std", "training_time", "inference_time"],
                                     columns=[classifier_name])
    for row in averaged_score_pd.index:
        for column in averaged_score_pd.columns:
            if row != "acc_std" and row != "inference_time" and row != "training_time":
                averaged_score_pd.loc[row][column] = np.mean(scores[row]["test"])
            elif row == "acc_std":
                averaged_score_pd.loc[row][column] = np.std((np.array(scores["accuracy"]["test"])*100).tolist())
            elif row == "training_time":
                averaged_score_pd.loc[row][column] = np.average(log_training_duration)
            elif row == "inference_time":
                averaged_score_pd.loc[row][column] = inference_time

    if classifier_name == CLASSIFIERS[0]:
        if os.path.exists(classifiers_comparison_log_dir):
            os.remove(classifiers_comparison_log_dir)
        _ = create_directory(MODELS_COMP_LOG_DIR)
        # save Averaged_CV_scores to CSV
        pd.DataFrame(["Averaged_CV_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header=False,
                                                    mode='a+')
        averaged_score_pd.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
    else:
        # add averaged_scores of new classifier
        saved_averaged_scores = pd.read_csv(classifiers_comparison_log_dir, skiprows=1, nrows=9, header=0, index_col=0)
        saved_averaged_scores = pd.concat([saved_averaged_scores, averaged_score_pd], axis=1)

        os.remove(classifiers_comparison_log_dir)
        pd.DataFrame(["Averaged_CV_scores"]).to_csv(classifiers_comparison_log_dir, index=False, header=False,
                                                    mode='a+')
        saved_averaged_scores.to_csv(classifiers_comparison_log_dir, index=True, mode='a+')
        
def log_save_CV_scores(log_training_duration, inference_time, scores, label2act, ACT_LABELS,\
                                  nb_classes, SNR, CV_SPLITS, y_test, LOG_DIR,\
                                      MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name):
        
        # log the test scores of every cross validation fold
        log_every_CV_score(logger, log_training_duration, scores, label2act, nb_classes, SNR, CV_SPLITS)
        # log the testing scores by averaging the predicts of all CV folds
        log_averaged_CV_scores(logger, log_training_duration, scores, label2act, nb_classes, SNR, CV_SPLITS)

        # Plot comfusion matrix
        plot_confusion_matrix(
            cms=scores["confusion_matrix"],
            labels=ACT_LABELS,
            path=f"{LOG_DIR}/comfusion_matrix.png",
        )

        save_classifiers_comparison(MODELS_COMP_LOG_DIR, CLASSIFIERS, classifier_name, scores, y_test,
                                    SNR, log_training_duration, inference_time)
