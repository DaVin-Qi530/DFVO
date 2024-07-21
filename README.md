# DFVO
DFVO: Learning Darkness-free Visible and Infrared Image Disentanglement and Fusion All at Once
## Network Structure
![framework](https://github.com/DaVin-Qi530/DFVO/blob/master/Figure/Framework.jpg)
The overall architecture of our method. The parallel cascaded tasks include the infrared image-reconstruction task, illumination disentanglement task, and image fusion task. (a) The specific structure of the Details-Extraction Module, which aims to capture high-frequency features from the source images. (b) The architecture of the Hyper Cross-Attention Module, which is designed to obtain the low-frequency features.
## LCFE Architecture
![module](https://github.com/DaVin-Qi530/DFVO/blob/master/Figure/Modules.jpg)
(a) The visual results of iteration process in the Details-Extraction Module. (b)The interaction details of the Hyper Cross-Attention Module.

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
