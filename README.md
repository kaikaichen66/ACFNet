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
│   ├── ACFNet_PVEL-AD.pt
│   └── ACFNet_PV-Multi-Defect.pt
│
├── train.py
├── valid.py
├── requirements.txt
├── CITATION.cff
├── LICENSE
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

The specific CUDA version can be adjusted according to the available GPU and PyTorch installation.

## 3. Datasets

This study uses two publicly available photovoltaic electroluminescence (EL) image datasets. The original datasets are not redistributed in this repository. Please obtain them from their respective original sources:

- **PVEL-AD:** https://github.com/binyisu/PVEL-AD
- **PV-Multi-Defect:** https://github.com/CCNUZFW/PV-Multi-Defect

The dataset configurations are provided in:

```text
datasets/
├── PVEL-AD.yaml
└── PV-Multi-Defect.yaml
```

### PVEL-AD

The subset used in this study contains 4,975 images with the following split:

```text
Train: 3977
Valid: 495
Test: 503
```

Seven defect categories are used:

```text
black_core
crack
finger
horizontal_dislocation
short_circuit
star_crack
thick_line
```

The corresponding configuration file is:

```text
datasets/PVEL-AD.yaml
```

### PV-Multi-Defect

The dataset used in this study contains 1,108 images with the following split:

```text
Train: 884
Valid: 112
Test: 112
```

Five defect categories are used:

```text
broken
hot_spot
black_border
scratch
no_electricity
```

The corresponding configuration file is:

```text
datasets/PV-Multi-Defect.yaml
```

After downloading the datasets, organize the image and label directories according to the paths specified in the corresponding YAML files.

## 4. Dataset Paths

The dataset YAML files use relative paths rather than the authors' local absolute paths.

For example, the PV-Multi-Defect configuration uses:

```yaml
path: datasets/PV-Multi-Defect

train: images/train
val: images/val
test: images/test
```

The expected directory structure is:

```text
datasets/
├── PVEL-AD.yaml
├── PV-Multi-Defect.yaml
└── PV-Multi-Defect/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

The same principle applies to PVEL-AD. If the datasets are stored in a different location, the `path` field in the corresponding YAML file can be adjusted accordingly.

## 5. Model Configurations

The repository provides the configurations used in the ablation experiments:

```text
configs/
├── baseline.yaml
├── SCMB.yaml
├── ACF.yaml
├── SCE.yaml
├── SCMB_ACF.yaml
├── SCMB_SCE.yaml
├── ACF_SCE.yaml
└── ACFNet.yaml
```

These configurations correspond to:

```text
Baseline
Baseline + SCMB
Baseline + ACF
Baseline + SCE
Baseline + SCMB + ACF
Baseline + SCMB + SCE
Baseline + ACF + SCE
Baseline + SCMB + ACF + SCE (ACFNet)
```

The corresponding configuration file can be selected in `train.py` when reproducing the individual experiments.

## 6. Training

After preparing the environment and datasets, run:

```bash
python train.py
```

The training script specifies the model configuration, dataset configuration, image size, training epochs, batch size, optimizer, random seed, and other training parameters used in the experiments.

For the complete ACFNet model, use:

```text
configs/ACFNet.yaml
```

The dataset configuration can be selected according to the target dataset:

```text
datasets/PVEL-AD.yaml
datasets/PV-Multi-Defect.yaml
```

## 7. Validation

The provided `valid.py` script is used for model evaluation.

Run:

```bash
python valid.py
```

The validation script uses the specified model configuration, dataset configuration, and trained weights.

The provided pretrained weights are:

```text
weights/ACFNet_PVEL-AD.pt
weights/ACFNet_PV-Multi-Defect.pt
```

## 8. Pretrained Weights

Pretrained ACFNet weights are provided for both datasets:

```text
weights/
├── ACFNet_PVEL-AD.pt
└── ACFNet_PV-Multi-Defect.pt
```

The PV-Multi-Defect model can be loaded using:

```python
from ultralytics import YOLO

model = YOLO("weights/ACFNet_PV-Multi-Defect.pt")
```

The PVEL-AD model can be loaded using:

```python
from ultralytics import YOLO

model = YOLO("weights/ACFNet_PVEL-AD.pt")
```

## 9. Reproduction

A reproduction procedure is provided below.

Clone the repository:

```bash
git clone https://github.com/kaikaichen66/ACFNet.git
cd ACFNet
```

For the version corresponding to the manuscript, use the specified repository commit:

```bash
git checkout COMMIT_HASH
```

Create the environment:

```bash
conda create -n acfnet python=3.9.23
conda activate acfnet
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download the original datasets from their respective repositories and organize them according to the provided dataset YAML files.

Check:

```text
datasets/PVEL-AD.yaml
datasets/PV-Multi-Defect.yaml
```

Then run the training script:

```bash
python train.py
```

For evaluation using the provided model weights:

```bash
python valid.py
```

The ablation experiments can be reproduced by selecting the corresponding configuration files in:

```text
configs/
```

## 10. Data Availability

The datasets used in this study are publicly available from their original repositories:

**PVEL-AD**

https://github.com/binyisu/PVEL-AD

**PV-Multi-Defect**

https://github.com/CCNUZFW/PV-Multi-Defect

The ACFNet source code, model configurations, dataset configurations, pretrained weights, and reproduction instructions are available at:

https://github.com/kaikaichen66/ACFNet

The exact repository commit corresponding to the manuscript is:

```text
COMMIT_HASH
```

The repository provides executable training and validation scripts:

```bash
python train.py
python valid.py
```

The original datasets are not redistributed in this repository. Users should obtain them from the respective original sources and follow their applicable licenses and terms of use.

## 11. License

This repository is released under the license specified in:

```text
LICENSE
```

The datasets used in this study are third-party public datasets. Please refer to their original repositories for the corresponding dataset licenses and terms of use.
