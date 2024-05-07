# Neural Networks & Deep Learning

<img src="../assets/img/neural networks.png" alt="neural networks">

## Introduction

### Neural Networks

Neural Networks, also known as Multilayer Perceptrons (MLP), represent a type of supervised machine learning model. With a structure loosely similar to the human brain, Neural Networks aim to capture non-linear relationships, in contrast to perceptrons, which function as linear classifiers.

### Machine Learning vs Deep Learning

## Intuition

The perceptron aims to learn a linear classifier represented by the equation `Wx + b`. In kernelization, the goal is to transform `x` into a higher-dimensional space `ϕ(x)`, making the data linearly separable. Thus, in kernelized perceptron, the equation becomes `W⋅ϕ(x) + b`. However, it is important to note that for `ϕ(x)`, we need to craft a well defined inner product. 

Unlike kernelization, neural networks learn a transformation and classifier simultaneously. It first perform an affine transformation on the input vector `x` and then apply a classifier. The equation for neural networks is `W⋅ϕ(x) + b`, where `ϕ(x)` is the result of applying a non-linear activation function `σ(z)` to the affine transformation `Ax+c`. Here, `A` and `c` perform the mapping, and and all parameters `A`, `c`, `W`, and `b` are learned during training.

<img src="../assets/img/neural-network-intuition-1.jpeg" alt="neural-network-intuition-1">

To understand what the neural network is learning, we can take an example of a neural network with one hidden layer and ReLU activation function

Consider a 


Another way to understand what Nerual Networks are learning is to look at 3B1B amazing visualzaion on neural networks. 

### SGD