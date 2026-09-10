# LADDER
<<<<<<< HEAD
=======
This is the project of manuscript Learnable Wavelet Decomposition with Adaptive Routing for Noise-Robust Rotating Machinery Fault Diagnosis, Ye Zhang, Aosheng Tian, Boyang Li, Longguang Wang, Hanyun Wang, and Yulan Guo. **Our team will release more interesting works and applications on time series analysis. Please keep following our repository.**
>>>>>>> 699f8c67dc2dec68727358c580e7615df2091af8

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-GPU%20required-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

Official PyTorch implementation of **“Learnable Wavelet Decomposition with Adaptive Routing for Noise-Robust Rotating Machinery Fault Diagnosis.”**

<<<<<<< HEAD
LADDER is a data-adaptive time-frequency model for fault diagnosis from noisy vibration signals. It combines learnable lifting-wavelet decomposition with Gumbel-Softmax routing so that the network can select which subbands should be decomposed further and at which level the representation should be read out.

![image-20260910202055552](/home/zhangye/snap/typora/118/.config/Typora/typora-user-images/image-20260910202055552.png)
=======
This paper proposes an **L**earnable w**A**velet **D**ecomposition network with a**D**aptiv**E** **R**outing* (**LADDER**) for rotating machinery fault diagnosis under noisy environment. The experimental results on three public datasets with  different noise intensities demonstrate the effectiveness of  the proposed method. The contributions of this paper are summarized as follows:

- We propose LADDER, a learnable wavelet decomposition network for noise-robust rotating machinery fault diagnosis, which comprises a lifting wavelet decomposition module and an adaptive decomposition routing.
- We introduce a lifting wavelet decomposition module with learnable prediction and update operators, thereby improving the data adaptivity of the decomposition filters at each level.
- We construct a wavelet decomposition routing that adaptively selects subbands for further decomposition and determines a sample-specific decomposition readout level, enhancing fault-noise separability.
- Extensive experiments across 6 public datasets and 6 noise levels demonstrate the superior noise robustness of LADDER, while ablation and interpretability analyses further validate its effectiveness.
>>>>>>> 699f8c67dc2dec68727358c580e7615df2091af8

> **Repository status:** research code accompanying a manuscript under review.

## Method Overview

This paper proposes an **L**earnable w**A**velet **D**ecomposition network with a**D**aptiv**E** **R**outing* (**LADDER**) for rotating machinery fault diagnosis under noisy environment. The  experimental results on three public datasets with  different noise  intensities demonstrate the effectiveness of  the proposed method. The  contributions of this paper are summarized as follows:

- We propose LADDER, a learnable wavelet decomposition network for noise-robust rotating machinery fault diagnosis, which comprises a lifting wavelet decomposition module and an adaptive decomposition routing.
- We introduce a lifting wavelet decomposition module with learnable prediction and update operators, thereby enabling data-adaptive decomposition at each level.
- We construct a wavelet decomposition routing that adaptively selects subbands for further decomposition and determines a sample-specific  decomposition readout level, enhancing the distinction between fault and noise components.
- Extensive experiments across 6 public datasets and 6 noise levels demonstrate the superior noise robustness of LADDER, while ablation and  interpretability analyses further validate its effectiveness.

## Paper-Reported Results

The following five-fold cross-validation accuracies are reported in the manuscript. Gaussian white noise is added independently at each matched train/test SNR.

| Dataset | -10 dB | -8 dB | -6 dB | -4 dB | -2 dB | 0 dB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CWRU | 89.33 | 94.30 | 98.23 | 99.17 | 99.73 | 99.83 |
| XJTU | 80.00 | 90.89 | 96.41 | 98.91 | 99.53 | 99.90 |
| SEU | 78.10 | 80.87 | 85.27 | 94.83 | 96.17 | 96.73 |
| MFPT | 97.12 | 98.51 | 98.52 | 98.65 | 99.50 | 99.91 |
| PU | 61.45 | 68.52 | 74.74 | 76.40 | 81.89 | 89.39 |
| IMS | 71.03 | 73.95 | 77.34 | 80.45 | 83.19 | 86.36 |

Across all 36 dataset-SNR settings, LADDER achieves a reported mean accuracy of **88.75%** and ranks first in 32 settings. The model contains approximately **98.24K parameters**.

## Repository Structure

```text
.
├── README.md
├── environment.yml
├── datasets/
└── src/
    ├── main.py
    ├── classifiers/
    │   └── compare/
    │       ├── LADDER.py
    │       └── ...
    └── utils/
        ├── constants.py
        ├── utils.py
        ├── CWRU/
        ├── XJTU/
        ├── SEU/
        ├── MFPT/
        ├── PU/
        ├── IMS/
        └── DWT_IDWT/
```

## Requirements

| Package      | Version |
| ------------ | ------- |
| Python       | 3.8.16  |
| PyTorch      | 2.4.1   |
| NumPy        | 1.23.5  |
| pandas       | 2.0.0   |
| SciPy        | 1.10.1  |
| scikit-learn | 1.2.2   |
| Matplotlib   | 3.7.1   |
| Seaborn      | 0.12.2  |
| tqdm         | 4.65.0  |
| gdown        | 5.2.2   |

## Datasets

All datasets can be **automatically** downloaded and extracted under the repository-level `datasets/` directory by the code. If a dataset fails to download, simply rerun  the code to resume from where it left off.

| CLI name | Dataset | Classes | Window | Samples used in paper | Source |
| --- | --- | ---: | ---: | ---: | --- |
| `CWRU_10` | Case Western Reserve University | 10 | 1024 | 3,000 | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) |
| `XJTU` | XJTU-SY | 15 | 1024 | 1,920 | [XJTU-SY dataset page](https://biaowang.tech/xjtu-sy-bearing-datasets/) |
| `SEU` | Southeast University gearbox dataset | 9 | 1024 | 3,000 | [Mechanical datasets](https://github.com/cathysiyu/Mechanical-datasets) |
| `MFPT` | Machinery Failure Prevention Technology | 3 | 1024 | 5,421 | [MFPT bearing fault data](http://www.mfpt.org/fault-data-sets/) |
| `PU` | Paderborn University | 14 | 2048 | 15,684 | [PU Bearing Data Center](https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter/data-sets-and-download) |
| `IMS` | IMS, University of Cincinnati | 2 | 1024 | 14,600 | [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings) |

Expected directory layout:

```text
datasets/
├── CWRU_10/
│   └── *.mat
├── XJTU/
│   ├── 35Hz12kN/
│   │   ├── Bearing1_1/
│   │   └── ...
│   ├── 37.5Hz11kN/
│   └── 40Hz10kN/
├── SEU/
│   ├── bearingset/
│   │   └── *.csv
│   └── gearset/
│       └── *.csv
├── MFPT/
│   ├── baseline_1.mat
│   ├── OuterRaceFault_1.mat
│   ├── InnerRaceFault_vload_1.mat
│   └── ...
├── PU/
│   ├── K001/
│   ├── KA04/
│   ├── KA15/
│   ├── KA16/
│   ├── KA22/
│   ├── KA30/
│   ├── KB23/
│   ├── KB24/
│   ├── KB27/
│   ├── KI04/
│   ├── KI16/
│   ├── KI17/
│   ├── KI18/
│   └── KI21/
└── IMS/
    ├── 2nd_test/
    └── 3rd_test/
```

Keep the original filenames and directory names because several loaders infer labels or chronological order directly from them. In particular:

- CWRU `.mat` files must be placed directly in `datasets/CWRU_10/`; the loader uses drive-end variables containing `DE_time`.
- XJTU CSV files must contain a `Horizontal_vibration_signals` column. The loader uses the last four chronologically numbered files from each bearing.
- SEU expects the exact case-sensitive filenames used by the original `bearingset` and `gearset` archives.
- IMS uses Bearing 1 from `2nd_test` and `3rd_test`; the first and last 10% of chronologically sorted files are labeled normal and faulty, respectively.
- The current PU loader implements the 14 labels listed in `src/utils/constants.py`, with `K001` as the healthy category.

## Training LADDER

Run commands from the `src/` directory so that local imports and output paths resolve correctly.

```bash
cd src
```

The code automatically trains and evaluates each requested dataset at six SNRs:

```text
-10, -8, -6, -4, -2, 0 dB
```

It also performs stratified five-fold cross-validation. The paper-specific commands are therefore best run separately for each dataset because learning rate and batch size are global command-line arguments.

```bash
# CWRU
python main.py --PATTERN TRAIN --DATASETS CWRU_10 --CLASSIFIERS_all LADDER \
    --BATCH_SIZE 64 --EPOCH 200 --LR 0.01 --CV_SPLITS 5 --test_split 3

# XJTU
python main.py --PATTERN TRAIN --DATASETS XJTU --CLASSIFIERS_all LADDER \
    --BATCH_SIZE 64 --EPOCH 200 --LR 0.001 --CV_SPLITS 5 --test_split 3

# SEU
python main.py --PATTERN TRAIN --DATASETS SEU --CLASSIFIERS_all LADDER \
    --BATCH_SIZE 128 --EPOCH 200 --LR 0.01 --CV_SPLITS 5 --test_split 3

# MFPT
python main.py --PATTERN TRAIN --DATASETS MFPT --CLASSIFIERS_all LADDER \
    --BATCH_SIZE 64 --EPOCH 200 --LR 0.01 --CV_SPLITS 5 --test_split 3

# PU
python main.py --PATTERN TRAIN --DATASETS PU --CLASSIFIERS_all LADDER \
    --BATCH_SIZE 64 --EPOCH 200 --LR 0.01 --CV_SPLITS 5 --test_split 3

# IMS
python main.py --PATTERN TRAIN --DATASETS IMS --CLASSIFIERS_all LADDER \
    --BATCH_SIZE 64 --EPOCH 200 --LR 0.001 --CV_SPLITS 5 --test_split 3
```

The `--test_split` argument controls inference-time chunking of a dataset to reduce GPU memory use; it is not the train/test split ratio.

## Evaluation from Saved Checkpoints

After a complete training run, evaluate the saved best-validation checkpoints with the same dataset, classifier, and cross-validation configuration:

```bash
python main.py --PATTERN TEST --DATASETS CWRU_10 --CLASSIFIERS_all LADDER \
    --BATCH_SIZE 64 --LR 0.01 --CV_SPLITS 5 --test_split 3
```

`TEST` mode expects `best_validation_model.pkl` and `score.csv` to exist for every requested fold and SNR. It does not download pretrained LADDER checkpoints.

## Command-Line Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--PATTERN` | `TRAIN` | Run mode: `TRAIN` or `TEST`. |
| `--DATASETS` | `CWRU_10 XJTU SEU` | One or more dataset identifiers. |
| `--CLASSIFIERS_all` | `LADDER` | One or more classifier identifiers. |
| `--BATCH_SIZE` | `64/128` | Training batch size. |
| `--EPOCH` | `200` | Number of training epochs; the paper uses 200. |
| `--LR` | `0.01/0.001` | Initial learning rate. |
| `--CV_SPLITS` | `5` | Number of stratified cross-validation folds. |
| `--test_split` | `3` | Number of inference chunks used to limit GPU memory. |

`src/main.py` currently sets `CUDA_VISIBLE_DEVICES="0"`. Edit that setting if a different GPU should be exposed.

## Comparison Models

The following paper baselines are available through `--CLASSIFIERS_all`:

```text
WDCNN, MCNN_LSTM, DRSN_CW, LiftingNet, WaveletKernelNet,
Wavelet_SANet, SoftFFRNet, AMCW_DFFNSA, MFSFormer, NPFormer,
MSDC_DenseTCN, ANC_Net, UniFault, BearLLM, Mantis
```

Multiple models can be requested in one command, for example:

```bash
python main.py --PATTERN TRAIN --DATASETS CWRU_10 \
    --CLASSIFIERS_all LADDER WDCNN DRSN_CW \
    --BATCH_SIZE 64 --EPOCH 200 --LR 0.01 --CV_SPLITS 5
```

Foundation-model baselines may download or require their own pretrained assets and can have substantially higher storage and GPU-memory requirements. Refer to the configuration variables near the top of their corresponding source files before running them.

## Outputs

Training and evaluation artifacts are written under `src/`:

```text
src/
├── saved_model/<dataset>/<classifier>/Cross_Validation_<fold><SNR>/
│   ├── init_model.pkl
│   ├── best_validation_model.pkl
│   ├── last_model.pkl
│   ├── history.csv
│   ├── history.png
│   └── score.csv
└── logs/<dataset>/<classifier>/<classifier>-<timestamp>/
    ├── <classifier>-<timestamp>.log
    └── comfusion_matrix.png
```

Aggregated classifier-comparison CSV files are stored in dataset-specific subdirectories of `src/logs/`.

## Reproducibility Notes

- Gaussian noise generation uses NumPy seed `66` in the dataset preprocessors.
- Cross-validation uses `StratifiedKFold(..., shuffle=True, random_state=66)`.
- The training/validation split uses `random_state=6` and preserves class proportions.

## Contact

For questions about the paper or code, please contact:

- Ye Zhang: `zhangy2658@mail.sysu.edu.cn, yundazhangye@163.com`

<<<<<<< HEAD
=======
  
>>>>>>> 699f8c67dc2dec68727358c580e7615df2091af8
