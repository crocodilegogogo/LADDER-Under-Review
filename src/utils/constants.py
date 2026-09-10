import os
import argparse

def parse_args():
    # The training options
      parser = argparse.ArgumentParser(description='LADDER')
      
      parser.add_argument('--PATTERN', type=str, default='TRAIN',
                           help='pattern: TRAIN, TEST')
      parser.add_argument('--DATASETS', nargs='+', default=['CWRU_10'],  # default=['CWRU_10','XJTU','SEU', 'MFPT','PU','IMS']
                           help='dataset name')
      parser.add_argument('--CLASSIFIERS_all', nargs='+', default=['LADDER'],
                          help='classifier name')
      # default=['LADDER', 'AMCW_DFFNSA', 'ANC_Net', 'DRSN_CW', 'LiftingNet', 'MCNN_LSTM', 'MFSFormer','MSDC_DenseTCN', 'NPFormer', \
      #                    'Resnet18', 'SoftFFRNet', 'Wavelet_SANet', 'WaveletKernelNet', 'WDCNN','Mantis','BearLLM','Mantis'],
      parser.add_argument('--BATCH_SIZE', type=int, default=64,
                          help='training batch size: 64/128')
      parser.add_argument('--EPOCH', type=int, default=200,
                          help='training epoches: 200')
      parser.add_argument('--LR', type=float, default=0.01,
                          help='learning rate: 0.01/0.001')
      parser.add_argument('--CV_SPLITS', type=int, default=5,
                          help='Cross Validation SPLITS: 5')
      parser.add_argument('--test_split', type=int, default=3,
                          help='the testing dataset is seperated into test_split pieces in the inference process')
      
      args = parser.parse_args()
      return args

def get_CWRU_dataset_param(CUR_DIR, dataset_name):
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '//datasets//CWRU_10//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison1//'
    Fault_LABELS = ["Normal","7Ball", "7innerRace", "7outerRace6",
                  "14Ball", "14innerRace", "14outerRace6",
                  "21Ball", "21innerRace", "21outerRace6", ]
    FaultID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    nb_classes = len(FaultID)
    WINDOW_SIZE = 1024
    OVERLAP = 1024  # the overlap of siliding window
    INPUT_CHANNEL = 1
    SNR = 5

    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes

def get_SEU_dataset_param(CUR_DIR, dataset_name):
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '//datasets//SEU//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison//'
    Fault_LABELS = ["Chipped_20_0","Chipped_30_2",
                    "Health_20_0", "Health_30_2",
                    "Miss_20_0", "Miss_30_2",
                    "Root_20_0", "Root_30_2",
                    "Surface_20_0"]
    FaultID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    nb_classes = len(FaultID)
    WINDOW_SIZE = 1024
    OVERLAP = 1024
    INPUT_CHANNEL = 1
    SNR = 5

    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes

def get_XJTU_dataset_param(CUR_DIR, dataset_name):
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '//datasets//XJTU//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison//'
    Fault_LABELS = ["Bearing 1_1", "Bearing 1_2",
                    "Bearing 1_3", "Bearing 1_4",
                    "Bearing 1_5",
                    "Bearing 2_1", "Bearing 2_2",
                    "Bearing 2_3", "Bearing 2_4",
                    "Bearing 2_5",
                    "Bearing 3_1", "Bearing 3_2",
                    "Bearing 3_3", "Bearing 3_4",
                    "Bearing 3_5",
                   ]
    FaultID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    nb_classes = len(FaultID)
    WINDOW_SIZE = 1024
    OVERLAP = 1024
    INPUT_CHANNEL = 1
    SNR = 5
    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes

def get_PU_dataset_param(CUR_DIR, dataset_name):
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '//datasets//PU//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison//'
    
    Fault_LABELS = [
        # "Health", # K001, K002, K003, K004, K005, K006
        "K001",
        "KA04",
        "KA15",
        "KA16",
        "KA22",
        "KA30",
        "KB23",
        "KB24",
        "KB27",
        "KI04",
        "KI16",
        "KI17",
        "KI18",
        "KI21",
    ]
    
    FaultID = list(range(len(Fault_LABELS)))
    nb_classes = len(Fault_LABELS)
    
    WINDOW_SIZE = 2048
    OVERLAP = 1024  # 50% overlap, note: in this project OVERLAP usually means 'stride'
    INPUT_CHANNEL = 1
    SNR = 5

    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes

def get_IMS_dataset_param(CUR_DIR, dataset_name):
    """
    Configure IMS dataset parameters.
    Structure assumption:
      datasets/IMS/
          ├── 2nd_test/   (Contains many files)
          ├── 3rd_test/   (Contains many files)
    """
    (filepath, _) = os.path.split(CUR_DIR)
    # Point to the root directory of the IMS dataset
    DATA_DIR = filepath + '//datasets//IMS//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison//'
    
    # Binary classification: Normal (0) vs Fault (1)
    Fault_LABELS = ["Normal", "Fault"]
    FaultID = [0, 1]
    nb_classes = len(FaultID)
    
    WINDOW_SIZE = 1024
    OVERLAP = 2048  # Or No overlap (Stride = Window Size)
    INPUT_CHANNEL = 1
    SNR = 5

    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes

def get_Connecticut_Geer_dataset_param(CUR_DIR, dataset_name):
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '//datasets//Connecticut-Geer//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison//'
    Fault_LABELS = ["Condition_1", "Condition_2", "Condition_3", "Condition_4", "Condition_5",
                    "Condition_6", "Condition_7", "Condition_8", "Condition_9"]
    FaultID = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    nb_classes = len(FaultID)
    WINDOW_SIZE = 1024
    OVERLAP = 512
    INPUT_CHANNEL = 1
    SNR = 5

    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes

def get_FEMTO_ST_dataset_param(CUR_DIR, dataset_name):
    # dataset_name is FEMTO-ST (from args)
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = filepath + '//datasets//FEMTO-ST-bearing-faults//'
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison//'
    
    Fault_LABELS = [
        "Normal",
        "Progressive", 
        "Fault"
    ]
    FaultID = [0, 1, 2]
    nb_classes = len(FaultID)
    WINDOW_SIZE = 2048
    OVERLAP = 0
    INPUT_CHANNEL = 1
    SNR = 5 # or whatever default
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes

def get_MFPT_dataset_param(CUR_DIR, dataset_name):
    (filepath, _) = os.path.split(CUR_DIR)
    # Ensure your MFPT dataset is placed in this path
    DATA_DIR = filepath + '//datasets//MFPT//' 
    MODELS_COMP_LOG_DIR = CUR_DIR + '//logs//' + dataset_name + '//classifiers_comparison//'
    
    # Define 3 types of fault labels
    Fault_LABELS = ["Normal", "OuterRace", "InnerRace"]
    FaultID = [0, 1, 2]
    nb_classes = len(FaultID)
    
    # Recommended parameters
    WINDOW_SIZE = 1024 # or 2048
    OVERLAP = 512      # 50% overlap
    INPUT_CHANNEL = 1
    SNR = 'False'      # or set to a specific number string, e.g., '5'

    return DATA_DIR, MODELS_COMP_LOG_DIR, Fault_LABELS, FaultID, \
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, SNR, nb_classes


def create_classifier(dataset_name, classifier_name, INPUT_CHANNEL,
                      data_length, nb_classes):

##################################

    if classifier_name == 'Resnet18':
        from classifiers.compare import Resnet18

        return Resnet18.resnet18(nb_classes, pretrained=False), Resnet18

    if classifier_name == 'WDCNN':
        from classifiers.compare import WDCNN

        return WDCNN.WDCNN(3,nb_classes,dataset_name), \
               WDCNN

    if classifier_name == 'MCNN_LSTM':
        from classifiers.compare import MCNN_LSTM

        return MCNN_LSTM.MCNN_LSTM(nb_classes), \
               MCNN_LSTM

    if classifier_name == 'DRSN_CW':
        from classifiers.compare import DRSN_CW
        return DRSN_CW.resnet18(nb_classes), DRSN_CW

    if classifier_name == 'LiftingNet':
        from classifiers.compare import LiftingNet

        return LiftingNet.LiftingNet(1, 3, nb_classes), \
               LiftingNet

    if classifier_name == 'WaveletKernelNet':
        from classifiers.compare import WaveletKernelNet

        return WaveletKernelNet.waveletkernelnet(nb_classes), \
               WaveletKernelNet

    if classifier_name == 'Wavelet_SANet':
        from classifiers.compare import Wavelet_SANet

        return Wavelet_SANet._4dwt_4(1, 9, 32, 32, 0.05, nb_classes, 1, 'db4'), \
               Wavelet_SANet

    if classifier_name == 'MSDC_DenseTCN':
        from classifiers.compare import MSDC_DenseTCN

        return MSDC_DenseTCN.MSDC_DenseTCN(nb_classes, 1), \
               MSDC_DenseTCN

    if classifier_name == 'ANC_Net':
        from classifiers.compare import ANC_Net

        return ANC_Net.ANC_Net(nb_classes, 1), \
               ANC_Net

    if classifier_name == 'MFSFormer':
        from classifiers.compare import MFSFormer

        return MFSFormer.MFSFormer(nb_classes), \
               MFSFormer

    if classifier_name == 'SoftFFRNet':
        from classifiers.compare import SoftFFRNet

        return SoftFFRNet.SoftFFRNet(nb_classes, data_length, INPUT_CHANNEL), \
               SoftFFRNet

    if classifier_name == 'NPFormer':
        from classifiers.compare import NPFormer

        return NPFormer.NPFormer(nb_classes, data_length, INPUT_CHANNEL), \
               NPFormer

    if classifier_name == 'AMCW_DFFNSA':
        from classifiers.compare import AMCW_DFFNSA

        return AMCW_DFFNSA.AMCW_DFFNSA(nb_classes, data_length), \
               AMCW_DFFNSA

    if classifier_name == 'UniFault':
        from classifiers.compare import UniFault
        if not os.getenv("UNIFAULT_PRETRAINED_CKPT") and os.getenv("UNIFAULT_AUTO_DOWNLOAD", "false").lower() not in {"1","true","yes"} and os.getenv("UNIFAULT_ALLOW_SCRATCH","false").lower() not in {"1","true","yes"}:
            print("Warning: UniFault requires a checkpoint unless UNIFAULT_ALLOW_SCRATCH=true.")
        return UniFault.UniFault(nb_classes, data_length, input_channels=INPUT_CHANNEL), UniFault

    if classifier_name == 'BearLLM':
        from classifiers.compare import BearLLM
        return BearLLM.BearLLM(nb_classes, data_length), BearLLM

    if classifier_name == 'Mantis':
        from classifiers.compare import Mantis
        return Mantis.MantisClassifier(data_length=data_length, numclass=nb_classes), Mantis

    if classifier_name == 'LADDER':
        from classifiers.compare import LADDER

        return LADDER.LADDER(dataset_name, nb_classes, data_length, first_conv_chnnl=16,
                                               kernel_size=3, no_bootleneck=False, average_mode="mode2",
                                               classifier="mode2", share_weights=False, simple_lifting=False,
                                               COLOR=True, regu_details=0.01, regu_approx=0.01, haar_wavelet=False), \
               LADDER