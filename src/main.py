import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils.constants import *
from utils.utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory

class Trainer(object):
    def __init__(self, args):
        
        # Initial
        self.args = args
        
        for dataset_name in args.DATASETS:
            for classifier_name in args.CLASSIFIERS_all:
                for i in range(6): # corresponding to different SNRs
                
                    # load raw data and important dataset param
                    X, Y, label2act, act2label, \
                    ACT_LABELS, ActID, MODELS_COMP_LOG_DIR, INPUT_CHANNEL, \
                    SNR, nb_classes = load_raw_data(dataset_name,CUR_DIR,i)
                    
                    # set logging settings
                    EXEC_TIME, LOG_DIR, MODEL_DIR, logger = logging_settings(classifier_name, CUR_DIR, dataset_name, SNR)
                    
                    # log the info of dataset and training params
                    log_dataset_info_training_parm(X, Y, ACT_LABELS, ActID, label2act, \
                                                   nb_classes, args.BATCH_SIZE, args.EPOCH, args.LR, \
                                                   args.CV_SPLITS, \
                                                   SNR)
                        
                    # K-fold cross validation
                    cv = StratifiedKFold(n_splits=args.CV_SPLITS, shuffle=True, random_state=66)
                    
                    # initialize the dict for logging variables
                    test_preds, models, scores, log_training_duration = initialize_saving_variables()
                    
                    # cycle CV folds
                    for fold_id, (train_index, test_index) in enumerate(cv.split(X, Y)):
                        # obtain the training+validation set and test set of each fold
                        X_tr, X_test, Y_tr, Y_test = trn_test_data_cv_split(X, Y, train_index, test_index)
                        # obtain the training and validation sets of each fold
                        X_tr, X_val, Y_tr, Y_val   = train_test_split(X_tr, Y_tr,
                                                                      test_size=0.2,
                                                                      random_state=6,
                                                                      stratify=Y_tr)
                        
                        # create classifier
                        classifier, classifier_func, classifier_parameter = create_cuda_classifier(dataset_name,
                                                                                                   classifier_name,
                                                                                                   INPUT_CHANNEL,
                                                                                                   X.shape[-1],
                                                                                                   nb_classes)
                        
                        # log train, validation and test data, log the network
                        log_trn_val_test_dataset_net_info(fold_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                                          Y_test, nb_classes, classifier_parameter, classifier)
                        
                        # train the network and save the best validation model
                        output_directory_models      = MODEL_DIR + '/Cross_Validation_' + str(fold_id) + str(SNR) +'/'
                        flag_output_directory_models = create_directory(output_directory_models)
                        # flag_output_directory_models = 'flag'
                        if args.PATTERN == 'TRAIN':
                            # for each CV fold, train the network and save the best validation model
                            print('Cross_Validation_' + str(fold_id) + ': start to train')
                            history, per_training_duration, log_training_duration = classifier_func.train_op(classifier,
                                                                                                             args.EPOCH,
                                                                                                             args.BATCH_SIZE, args.LR,
                                                                                                             X_tr, Y_tr,
                                                                                                             X_val, Y_val,
                                                                                                             output_directory_models,
                                                                                                             log_training_duration,
                                                                                                             args.test_split)
                        
                        else:
                            print('Already_done: ' + 'Cross_Validation_' + str(fold_id) + str(SNR) + 'pr')
                            # read the training duration of current Cross Validation
                            per_training_duration = pd.read_csv(output_directory_models + 'score.csv',
                                                                skiprows=1, nrows=1, header=None)[1][0]
                            log_training_duration.append(per_training_duration)
                        
                        # input: X_tr, X_val, X_test, output: pred_train, pred_valid, pred_test (the one hot predictions)
                        # save the metrics per CV
                        pred_train, pred_valid, pred_test, scores, inference_time = predict_tr_val_test(classifier,
                                                                                                        nb_classes,
                                                                                                        ACT_LABELS,
                                                                                                        X_tr, X_val, X_test,
                                                                                                        Y_tr, Y_val, Y_test,
                                                                                                        scores,
                                                                                                        per_training_duration,
                                                                                                        fold_id,
                                                                                                        output_directory_models,
                                                                                                        args.test_split)
                    
                    # log and save the testing CV scores, plot comfusion matrix
                    log_save_CV_scores(log_training_duration, inference_time, scores, label2act, ACT_LABELS,\
                                       nb_classes, SNR, args.CV_SPLITS, Y_test, LOG_DIR,\
                                       MODELS_COMP_LOG_DIR, args.CLASSIFIERS_all, classifier_name)

def main(args):
    trainer = Trainer(args)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)


