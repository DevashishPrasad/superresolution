# Blind single image super resolution

## Introduction

Super-resolution (SR) has been studied and implemented since the 1990s because of its wide range of applications. It is used in Medical Imaging, CCTV surveillance, Gaming, and Astronomical Imaging, to name a few. Training practical SR models are typically challenging because these models tend to get biased toward training data distribution (type of degradation of low resolution (LR) images). Traditional SR models don’t generalize well to real-world unknown test time images. 

<img src="https://github.com/DevashishPrasad/superresolution/blob/main/pictures/1.png" />

<img src="https://github.com/DevashishPrasad/superresolution/blob/main/pictures/2.png" />

Recently, researchers have been paying more attention to making the SR models more robust such that they become invariant to the degradation process of the LR image input. It is known as the blind image SR task that aims to super-resolved LR images that result from an unknown degradation process and generate high resolution (HR) images. 

<img src="https://github.com/DevashishPrasad/superresolution/blob/main/pictures/3.png" />

In this project, I present a detailed comparative analysis of two recent state-of-the-art Blind and one older but prominent Non-blind SR method. All three methods were originally trained (by their authors) in different training and testing environments. And so, these pre-trained models cannot be compared directly, which is the primary motivation of this project. To compare these models fairly, in this project, I carry out detailed experiments of training and evaluating these models in a common training and testing setting. 

Specifically, in this repository I present a benchmark for comparing three prominent super-resolution models **EDSR vs UDRL vs KOALAnet**.

- EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution [link](https://arxiv.org/abs/1707.02921)
- UDRL: Unsupervised Degradation Representation Learning for Blind Super-Resolution [link](https://arxiv.org/abs/2104.00416)
- KOALAnet: KOALAnet: Blind Super-Resolution using Kernel-Oriented Adaptive Local Adjustment [link](https://arxiv.org/abs/2012.08103)

Both papers UDRL and KOALAnet share amazing results for blind image super resolution tasks. But, their training and testing methodologies differ a lot and we cannot conclude which model performs better than other. And this motivated me to create this project in which I carried out several experiments to compare these two models. I also compare UDRL and KOALAnet with a non-blind EDSR model and Bicubic interpolation baseline algorithm. Specifically I did the following - 

1. Create a common benchmark dataset and evaluated the pre-trained models released by the authors or found on internet.
2. Reproduced and trained the models from scratch and evaluated the trained models.

## Common dataset
For the common and fair evaluation, I use a test set published by KOALAnet which was generated using five benchmark datasets using different degradation parameters (of Equation y = (x ⊗ k) ↓ s + n) for different images randomly. I find this dataset the most appropriate because it was created using a complex degradation kernel of type an-isotropic. Such a difficult dataset benchmark will test all the models to their limit. I only use a 4x upscaling factor as it is more challenging than 2x, and the model needs to really perform well to upscale an image four times. Lastly, I use consistent metrics of PSNR and SSIM over whole images (instead of considering just a particular channel) across all models. All Github released codes are implemented using different and older versions of Deep Learning frameworks (EDSR=OpenCV 3.X, UDRL=Pytorch 1.1.0, and KOALAnet=TensorFlow 1.13). 

I evaluate these pre-trained model on evaluation dataset of KOALAnet and the following table shows the results of my evaluation.

<img src="https://github.com/DevashishPrasad/superresolution/blob/main/pictures/4.png" />

## Reproducing and training models from scratch
The previous pre-trained Github code models-based evaluation was not fair because each model was trained and evaluated differently. So, to train all models under the same scheme and dataset, I reproduce EDSR and UDRL from scratch in Pytorch and train them using a common dataset. Unfortunately, KOALAnet was written in TensorFlow 1.X, and because there were lots of complications in the code, I was not able to reproduce it in Pytorch. But, the authors did publish the training and evaluation code, so I use the same to train the model from scratch and change the evaluation code according to our dataset and metrics. All models had their own different original training strategies. I remove this inconsistency by using a common training strategy i.e same dataset, degradation parameters (same training data distribution), augmentations, a learning rate scheduler, etc. across all models. I also keep the number of parameters and number of epochs about the same for all models. The following table shows the results of our evaluation of these models trained from scratch.


Table 1: PSNR and SSIM of models trained from scratch on 5 standard benchmark datasets. (upscaling
factor x4)

<img src="https://github.com/DevashishPrasad/superresolution/blob/main/pictures/5.png" />

## More information 

please read the paper in this repo - Blind_Image_Super_Resolution.pdf
