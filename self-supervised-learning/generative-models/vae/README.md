# Variational Autoencoder

## Introduction

Variational Autoencoders can be thought of as vanilla autoencoders modified for generation task. Autoencoders have the capability to represent input in a compressed latent space, and then we can interpolate in the latent space to discover new samples. However, for a generative model, we require the ability to sample. That is, I wish to generate a new image. In autoencoders, we cant create a random latent vector and expect it to work, beacause of holes in the latent space. the latent space of autoencoder was never designed to be used for sampling l;ater. Now, if we wish to sample, we expect the ;latent space to be a probability distribution, and that too from some known family for each sampling. And what can be easier than stanmdard gaussian.

In other words, vae are generative siblings of autoencoder, where the latnt space is now a probability distribution.

Naturally, because VAEs are expected to align the latents as a distribution, it hurts the representatio0n somewhat. But then it gives us a generational capablity. 