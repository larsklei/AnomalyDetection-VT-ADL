# Unsupervised Anomaly Detection with VisionTransformers and Registers.
This repository contains a on-going private project to implement a VisionTransformer Encoder-Decoder approach to detect 
anomalies in images. The main idea is from the paper ["VT-ADL: A Vision Transformer Network for 
Image Anomaly Detection and Localisation" by Mishra, et al.](https://arxiv.org/abs/2104.10036).
Further we want to implement the ideas of registers from the paper ["Vision Transformer needs Registers" by Darcet et al.](https://arxiv.org/abs/2309.16588)
## Prerequisites
### Installation & Requirements
A detailed list of requirements can be found in `requirements.txt`. To install all required packages, type
```
pip install -r requirements.txt
```
We tested the code with Python 3.11.4 on a Macbook Pro with a M1 Pro chip. 
We expect the code to be running also on other systems, such as Windows and Linux distributions.
### Repository structure
This repository provides a implementation of a VisionTransformer Autoencoder, losses, etc. which have been discussed in
the paper by Mishra et al. The Gaussian Mixture Density Network has not been implemented yet. We plan to in the future.
- `vit_ae.model_utilities.loss_metric_utilities` provides the implementation of the loss function.
- `vit_ae.preprocess.preprocess_utilities` provides the code for the dataset generator.
- `vit_ae.train.train_utilities` provides methods to create models and optimizer.
- `vit_ae.custom_model` provides the implementation of the VisionTransformerAutoEncoder, VisionTransformerEncoder, PatchEmbedding as subclassed Keras models or layers.
- `vit_ae.experiments` provides the code for doing a train run.
### Dataset
We use the [MVTEC AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) for this project.
Here, it is **important** that the source directory has the same structure as the MVTEC dataset:
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
### Training and Evaluation
To start a simple run without hyperparameter optimization, you just need to run
```
python3 vit_ae/experiments/vt_ad_experiment.py [args]
```
We used [Click](https://click.palletsprojects.com/en/8.1.x/) to provide a CLI for passing parameters.

## Future Work
The repository is a still on-going project and still not finished. Here are some problems which will be addressed later:
- [ ] Currently, we use the L1-norm to create the grey mask for the prediction. We would prefer to use the Structural similarity measure to calculate the mask, 
        but the method `tensorflow.images.ssim` returns a local map with the wrong shape (see [the discussion here](https://github.com/tensorflow/tensorflow/issues/59067))
        We plan to re-write SSIM to being used with `evalute`-methods of the `keras.Model`-class
- [ ] Streamline the train run and implement the use of MLflow.
- [ ] Set up HPO and do serious evaluation run.