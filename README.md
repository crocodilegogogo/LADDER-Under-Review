# LADDER
This is the project of manuscript Learnable Wavelet Decomposition with Adaptive Routing for Noise-Robust Rotating Machinery Fault Diagnosis, Ye Zhang, Aosheng Tian, Boyang Li, Longguang Wang, Hanyun Wang, and Yulan Guo. **Our team will release more interesting works and applications on time series analysis. Please keep following our repository.**

![architechture](AWD-Net.png)

## Algorithm Introduction

This paper proposes an **L**earnable w**A**velet **D**ecomposition network with a**D**aptiv**E** **R**outing* (**LADDER**) for rotating machinery fault diagnosis under noisy environment. The experimental results on three public datasets with  different noise intensities demonstrate the effectiveness of  the proposed method. The contributions of this paper are summarized as follows:

- We propose LADDER, a learnable wavelet decomposition network for noise-robust rotating machinery fault diagnosis, which comprises a lifting wavelet decomposition module and an adaptive decomposition routing.
- We introduce a lifting wavelet decomposition module with learnable prediction and update operators, thereby improving the data adaptivity of the decomposition filters at each level.
- We construct a wavelet decomposition routing that adaptively selects subbands for further decomposition and determines a sample-specific decomposition readout level, enhancing fault-noise separability.
- Extensive experiments across 6 public datasets and 6 noise levels demonstrate the superior noise robustness of LADDER, while ablation and interpretability analyses further validate its effectiveness.

## Prerequisite

Tested on Ubuntu 22.04, with Python 3.8, PyTorch 1.12, CUDA 11.3, and 1x NVIDIA 2080Ti.

## Usage

On Ubuntu:

```
Click on main.py and run it. Adjust the hyperparameters in the constant file.
```

Or:

**(1) Train.**

```python
python main.py --PATTERN TRAIN --DATASETS CWRU_10 --CLASSIFIERS_all AWD_Net --BATCH_SIZE 64 --EPOCH 200 --LR 0.01 --CV_SPLITS 5 --test_split 20
```

**(2) Test.**

After the training process, we can use the following command for testing.

```
python main.py --PATTERN TEST --DATASETS CWRU_10 --CLASSIFIERS_all AWD_Net --test_split 20
```

## Other Explanations

Limited by storage space, we only upload the CWRU_10 dataset in the project.


  
