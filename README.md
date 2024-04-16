# DFVO
DFVO: Learning Darkness-free Visible and Infrared Image Disentanglement and Fusion All at Once
## Framework
![framework](https://github.com/DaVin-Qi530/DFVO/blob/master/Figure/Framework.jpg)
The architecture of our DFVO method. Dashed lines represent the data-flow for hidden i-stage, and solid lines represent the data-flow for hidden ii-stage. The feature extraction module serves both hidden tasks (illumination enhancement & image fusion) in a shared manner.
## FEM Architecture
![module](https://github.com/DaVin-Qi530/DFVO/blob/master/Figure/Modules.jpg)
The details of the modules. (left) Interaction details of the cross-attention module. (right) Specific blocks of the details-extraction module.

## About Code
This paper is currently under review. We will open-source it as soon as it is accepted for publication.

## Fusion Demo
![fusion_results1](https://github.com/DaVin-Qi530/DFVO/blob/master/Figure/Fusion.jpg)
Vision quality comparison of five SOTA fusion methods on the LLVIP dataset.

## Two-stage Fusion Demo
![fusion_results2](https://github.com/DaVin-Qi530/DFVO/blob/master/Figure/Fusion_2.jpg)
Vision quality comparison of two-stage fusion methods on the LLVIP dataset.

## Detection Results
![detect_results](https://github.com/DaVin-Qi530/DFVO/blob/master/Figure/Detect.jpg)
Detection performance of our fused images with four SOTA fusion results on the LLVIP dataset.
