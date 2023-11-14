# Unsupervised Anomaly Detection with VisionTransformers and Registers.
This repository contains a private project to implement a VisionTransformer Encoder-Decoder approach to detect 
anomalies in images. The main idea is from the paper "VT-ADL: A Vision Transformer Network for 
Image Anomaly Detection and Localisation" by Mishra, et al. ( https://arxiv.org/abs/2104.10036 ).
Further we want to implement the ideas of registers from the paper "Vision Transformer needs Registers" by Darcet et al. ( https://arxiv.org/abs/2309.16588 ).
## Work in Progress (To-Do List)
    [ ] Fix the get_config method for the custom model to load model correctly. 
    [ ] Implement a more sophisticated test step. This requires a new implementation of the SSIM score in tensorflow since
        the actual version does not return a correct local map (see https://github.com/tensorflow/tensorflow/issues/59067 )
    [ ] Implement hyperparameter script.
    [ ] Write Documentation.
## Prerequisites
### Installation
We recommend to set up a virtual environment. To install all required packages, type
```
pip install -r requirements.txt
```
### Dataset
We use the MVTEC AD dataset for this project. 
Here, it is important that the source directory has the same structure as follow:
```
    ├── bottle
    │   ├── ground_truth
    │   │   ├── broken_large
    │   │   ├── broken_small
    │   │   └── contamination
    │   ├── test
    │   │   ├── broken_large
    │   │   ├── broken_small
    │   │   ├── contamination
    │   │   └── good
    │   └── train
    │       └── good
    ...
```