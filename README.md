# ACFNet

**ACFNet: Asymmetric Cross-layer Fusion Network with Semantic Consistency Enhancement for Photovoltaic Defect Detection**

This repository provides the implementation of ACFNet, including the model configurations, dataset configurations, training and validation scripts, and pretrained weights used in the study.

## 1. Repository Structure

```text
ACFNet/
├── configs/
│   ├── baseline.yaml
│   ├── SCMB.yaml
│   ├── ACF.yaml
│   ├── SCE.yaml
│   ├── SCMB_ACF.yaml
│   ├── SCMB_SCE.yaml
│   ├── ACF_SCE.yaml
│   └── ACFNet.yaml
│
├── datasets/
│   ├── PVEL-AD.yaml
│   └── PV-Multi-Defect.yaml
│
├── weights/
│   └── ACFNet_PV-Multi-Defect.pt
│
├── train.py
├── valid.py
├── requirements.txt
└── README.md
```

The files in `datasets/` are dataset configuration files only. The original datasets are not redistributed in this repository.

## 2. Environment

The experiments were conducted using the following environment:

```text
Python       3.9.23
PyTorch      2.7.1
Torchvision  0.22.1
Ultralytics  8.3.57
CUDA         12.6
```

Create the environment and install the required packages:

```bash
conda create -n acfnet python=3.9.23
conda activate acfnet
pip install -r requirements.txt
```

## 3. Datasets

This study uses two publicly available photovoltaic electroluminescence (EL) image datasets: PVEL-AD and PV-Multi-Defect. The original datasets are not redistributed in this repository and should be obtained from their respective original sources.

The dataset configuration files are provided in:

```text
datasets/
├── PVEL-AD.yaml
└── PV-Multi-Defect.yaml

## 4. Model Configurations

The repository provides all configurations used in the ablation experiments:

```text
configs/baseline.yaml
configs/SCMB.yaml
configs/ACF.yaml
configs/SCE.yaml
configs/SCMB_ACF.yaml
configs/SCMB_SCE.yaml
configs/ACF_SCE.yaml
configs/ACFNet.yaml
```

These configurations correspond to the baseline model, individual modules, module combinations, and the complete ACFNet.

## 5. Training

After preparing the environment and datasets, run:

```bash
python train.py
```

The corresponding model and dataset configurations should be specified in `train.py` before training.

For the complete model, use:

```text
configs/ACFNet.yaml
```

## 6. Validation

Run:

```bash
python valid.py
```

The validation script uses the specified model configuration, dataset configuration, and trained weights.

## 7. Pretrained Weights

The pretrained ACFNet model trained on the PV-Multi-Defect dataset is provided in:

```text
weights/ACFNet_PV-Multi-Defect.pt
```

The model can be loaded using:

```python
from ultralytics import YOLO

model = YOLO("weights/ACFNet_PV-Multi-Defect.pt")
```

## 8. Reproduction

A complete reproduction procedure is summarized below:

```bash
git clone https://github.com/kaikaichen66/ACFNet.git
cd ACFNet

conda create -n acfnet python=3.9.23
conda activate acfnet

pip install -r requirements.txt
```

After downloading and preparing the datasets, verify the dataset paths in:

```text
datasets/PVEL-AD.yaml
datasets/PV-Multi-Defect.yaml
```

Then run:

```bash
python train.py
python valid.py
```

The ablation experiments can be reproduced by selecting the corresponding configuration files in `configs/`.

## 9. Data Availability

The datasets used in this study are not redistributed in this repository. Users should obtain the PVEL-AD and PV-Multi-Defect datasets from their respective original sources and follow their applicable licenses and terms of use.

The ACFNet source code, model configurations, dataset configurations, pretrained weights, and reproduction scripts are provided in this repository.

## 10. License

This repository is released under the MIT License. See the `LICENSE` file for details.

The datasets used in this study are third-party public datasets. Please refer to their original sources for the corresponding dataset licenses and terms of use.
