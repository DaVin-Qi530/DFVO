# DFVO
DFVO: Learning Darkness-free Visible and Infrared Image Disentanglement and Fusion All at Once 🚀
> [![arXiv](https://img.shields.io/badge/arXiv-Paper-blue.svg)](https://arxiv.org/abs/2505.04526)<br>
## Network Structure
![framework](https://github.com/DaVin-Qi530/DFVO/blob/master/Figures/Framework.jpg)
The overall architecture of our method. The parallel cascaded tasks include the infrared image-reconstruction task, illumination disentanglement task, and image fusion task. (a) The specific structure of the Details-Extraction Module, which aims to capture high-frequency features from the source images. (b) The architecture of the Hyper Cross-Attention Module, which is designed to obtain the low-frequency features.
## LCFE Architecture
![module](https://github.com/DaVin-Qi530/DFVO/blob/master/Figures/Modules.jpg)
(a) The visual results of iteration process in the Details-Extraction Module. (b)The interaction details of the Hyper Cross-Attention Module.

## About Code
### Recommended Environment
```
numpy=1.21.5
python=3.7.13
opencv=4.8.1
pytorch=1.11.0
rich=13.9.4
thop=0.1.1
```

### Dataset
The LLVIP dataset can be downloaded via the following link: [here](https://bupt-ai-cz.github.io/LLVIP/).
The KAIST dataset can be downloaded via the following link: [here](https://github.com/SoonminHwang/rgbt-ped-detection).
The MSRS dataset can be downloaded via the following link: [here](https://github.com/Linfeng-Tang/MSRS).
The SMOD dataset can be downloaded via the following link: [here](https://www.kaggle.com/datasets/zizhaochen6/sjtu-multispectral-object-detection-smod-dataset).

### To Train
python train.py --train_dataset yours_dataset --model yours_model

### To Eval
python eval.py\
using our pre-trained model

## Fusion Demo
![fusion_results1](https://github.com/DaVin-Qi530/DFVO/blob/master/Figures/Fusion.jpg)
Vision quality comparison of five SOTA fusion methods on the LLVIP dataset.

## Two-stage Fusion Demo
![fusion_results2](https://github.com/DaVin-Qi530/DFVO/blob/master/Figures/Fusion_2.jpg)
Vision quality comparison of two-stage fusion methods on the LLVIP dataset.

## Generalization Fusion Demo
![fusion_results3](https://github.com/DaVin-Qi530/DFVO/blob/master/Figures/Fusion_3.jpg)
Vision quality comparison of two-stage fusion methods on the MSRS, SMOD, and KAIST datasets.

## Detection Results
![detect_results](https://github.com/DaVin-Qi530/DFVO/blob/master/Figures/Detect.jpg)
Detection performance of our fused images with four SOTA fusion results on the LLVIP dataset.

## If this work is helpful to you, please cite it as:
