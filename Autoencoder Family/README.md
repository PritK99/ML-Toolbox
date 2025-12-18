# Autoencoder Family

<div style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <div style="text-align: center;">
    <img src="../assets/img/latent_space_visualization.png" alt="Abstraction" width="250">
    <br>
    <small><i>Visualizing latent vectors (Abstraction)</i></small>
  </div>

  <div style="text-align: center;">
    <img src="../assets/img/reconstruction_from_T-shirt_Bag.png" alt="Generation" height="75" width="300">
    <br>
    <small><i>Creating new samples (Generation)</i></small>
  </div>
</div>

## Introduction

Apart from tasks like classification, regression, clustering, etc., we are also interested in abstraction and generation. Abstraction means learning representations for our data. For example, we can convert an image into a feature vector. This idea of learning low-dimensional feature representation is useful for many tasks, such as visualization or similarity. Generation means creating new data samples such that they belong to the original data distribution. Autoencoders are one family of models that can do these tasks. While there are several flavors of autoencoders, we will focus on vanilla autoencoders and variational autoencoders (VAEs).

## Vanilla Autoencoders


<p align="center">
  <img src = "../assets/img/autoencoder.png" alt="autoencoders">
  <br>
  <small><i>Creating new samples (Generation)</i></small>
</p>

### Results

#### Abstraction 

<p align="center">
  <img src = "../assets/img/latent_space_visualization.png" alt="latent-space">
</p>

<p align="center">
  <img src = "../assets/img/reconstructed_samples_epoch_20.png" alt="reconstructed samples">
</p>

#### Generations

While autoencoders are not designed for generation, they too can give good generations sometimes. We look at the latent space and choose classes that are close to each other and try to generate a few images. These are some examples where the generations were really amazing:

<p align="center">
  <img src = "../assets/img/reconstruction_from_T-shirt_Bag.png" alt="interpolation of tshirt and bag">
  <br>
  <small><i>The generated output resembles both a t-shirt and a bag</i></small>
</p>

<p align="center">
  <img src = "../assets/img/reconstruction_from_Sandal_Sneaker_Ankle boot.png" alt="interpolation of sandals, sneakers and ankle boots">
</p>

### Stress Test 01: Random Rotations

We wanted to explore how the autoencoder behaves when its input is distorted. For each image, we first apply a random rotation of 0°, 90°, 180°, or 270°. The rotated image is then fed into the autoencoder, whose task is to reconstruct the original, unrotated image. Success in this task might require the model to learn rotation-invariant features. While I expected the autoencoder to struggle with undoing the rotations, the results are amazing. Below, we show the reconstruction outputs after the 1st epoch and after the 20th epoch.

<p align="center">
  <img src = "../assets/img/random_rotations_reconstructed_samples_epoch_1.png" alt="random-rotations">
  <br>
  <small><i>After 1st epoch</i></small>
</p>

<p align="center">
  <img src = "../assets/img/random_rotations_reconstructed_samples_epoch_20.png" alt="random-rotations">
  <br>
  <small><i>After 20th epoch</i></small>
</p>

Here, K denotes the multiple of 90 degree that was used to rotate the image. The rotations are random, even across epochs, but still the autoencoder has managed to reconstruct the original image.