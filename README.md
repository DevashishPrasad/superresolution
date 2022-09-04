# Blind single image super resolution

In this repository I present a benchmark for comparing three prominent super-resolution models **EDSR vs UDRL vs KOALAnet**.

- EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution [link](https://arxiv.org/abs/1707.02921)
- UDRL: Unsupervised Degradation Representation Learning for Blind Super-Resolution [link](https://arxiv.org/abs/2104.00416)
- KOALAnet: KOALAnet: Blind Super-Resolution using Kernel-Oriented Adaptive Local Adjustment [link](https://arxiv.org/abs/2012.08103)

Both papers UDRL and KOALAnet share amazing results for blind image super resolution tasks. But, their training and testing methodologies differ a lot and we cannot conclude which model performs better than other. And this motivated me to create this project in which I carried out several experiments to compare these two models. I also compare UDRL and KOALAnet with a non-blind EDSR model and Bicubic interpolation baseline algorithm. Specifically I did the following - 

1. Create a common benchmark dataset and evaluated the pre-trained models released by the authors or found on internet.
2. Reproduced and trained the models from scratch and evaluated the trained models.

## Common dataset
For the common and fair evaluation, I use a test set published by KOALAnet which was generated using five benchmark datasets using different degradation parameters (of Equation y = (x ⊗ k) ↓ s + n) for different images randomly. I find this dataset the most appropriate because it was created using a complex degradation kernel of type an-isotropic. Such a difficult dataset benchmark will test all the models to their limit. We only use a 4x upscaling factor as it is more challenging than 2x, and the model needs to really perform well to upscale an image four times. Lastly, we use consistent metrics of PSNR and SSIM over whole images (instead of considering just a particular channel) across all models. All Github released codes are implemented using different and older versions of Deep Learning frameworks (EDSR=OpenCV 3.X, UDRL=Pytorch 1.1.0, and KOALAnet=TensorFlow 1.13). 

## Reproducing and training models from scratch
The previous pre-trained Github code models-based evaluation was not fair because each model was trained and evaluated differently. So, to train all models under the same scheme and dataset, we reproduce EDSR and UDRL from scratch in Pytorch and train them using a common dataset. Unfortunately, KOALAnet was written in TensorFlow 1.X, and because there were lots of complications in the code, we were not able to reproduce it in Pytorch. But, the authors did publish the training and evaluation code, so we use the same to train the model from scratch and change the evaluation code according to our dataset and metrics. As we saw in section 3, all models had their own different original training strategies. We remove this inconsistency by using a common training strategy i.e same dataset, degradation parameters (same training data distribution), augmentations, a learning rate scheduler, etc. across all models. We also keep the number of parameters and number of epochs about the same for all models. The following table shows the results of our evaluation of these models trained from scratch.


Table 1: PSNR and SSIM of models trained from scratch on 5 standard benchmark datasets. (upscaling
factor x4)

<table>
<tr>
<td>Method</td> <td>No Params</td> <td>No Epochs</td> <td>Set 5</td> <td>Set 14</td> <td>BSD 100</td> <td>Urban 100</td> <td>Manga 109</td> <td>Div2K Val 100</td>
</tr>
<tr>
<td>Bicubic</td> <td>NA</td> <td>NA</td> <td>24.86/0.7105</td> <td>23.20/0.6240</td> <td>23.86/0.6030</td> <td>20.67/0.5774</td> <td>22.09/0.7183</td> <td>25.71/0.7184</td>
</tr>
<tr>
<td>EDSR+</td> <td>5,487,433</td> <td>300</td> <td>25.68/0.7393</td> <td>23.02/0.6347</td> <td>23.82/0.6287</td> <td>20.94/0.6041</td> <td>22.64/0.7303</td> <td>25.74/0.73060</td>
</tr>
<tr>
<td>UDRL</td> <td>5,969,283</td> <td>100+200</td> <td>20.70/0.5381</td> <td>19.99/0.4913</td> <td>21.50/0.5007</td> <td>19.22/0.4654</td> <td>18.26/0.5337</td> <td>22.65/0.5980</td>
</tr>
<tr>
<td>KOALAnet</td> <td>6,545,416</td> <td>50+50+110</td> <td>25.72/0.7752</td> <td>23.88/0.6721</td> <td>24.39/0.6529</td> <td>21.60/0.6515</td> <td>23.40/0.7839</td> <td>26.22/0.7592</td>
</tr>
</table>

