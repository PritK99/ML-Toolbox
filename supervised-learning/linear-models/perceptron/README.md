# Perceptron

<p align="center">
  <img src = "../../../assets/img/perceptron/visualization.gif" alt="perceptron visualization">
  <br>
  <small><i>Image source: https://commons.wikimedia.org/wiki/File:PerceptronSample.gif</i></small>
</p>

## Table of Contents

- [Perceptron](#perceptron)
  - [Introduction](#introduction)
  - [Algorithm](#algorithm)
    - [Why does the update rule work?](#why-does-the-update-rule-work)
    - [Proof that the Perceptron will converge](#proof-that-the-perceptron-will-converge)
  - [Usage](#usage)
  - [Results & Demo](#results--demo)

## Introduction

The perceptron is one of the earliest linear classification algorithms. Linear classifiers assume that there exists a hyperplane that can separate the data into different classes. The goal of the perceptron is therefore to learn a hyperplane $w^T x + b = 0$ such that all the data points are correctly classified. The fact that the data can be separated using a linear hyperplane is the knowledge, while the parameters of the hyperplane are learned from the data.

## Algorithm

A hyperplane is a subspace whose dimension is one less than that of the feature space. For example, in a 2D feature space, a hyperplane is simply a straight line, while in 3D it is a plane. A hyperplane can be defined using:

1) A weight vector w, which is normal (perpendicular) to the hyperplane, and
2) A bias term b, which controls the offset from the origin.

Hence, it can be written as $w^T x + b = 0$. But, this is as good as $W^T X = 0$ where `W` = `[w, b]` and `X` = `[x, 1]`. Because the bias term is absorbed into `W` now, we don't have to do additional bookkeeping for the bias term.

Now, we can use the following algorithm to learn the parameters for the hyperplane.

<p align="center">
  <img src = "../../../assets/img/perceptron/perceptron-algorithm.jpeg" alt="perceptron algorithm">
</p>

During inference, we just need to check the sign of $W^T X$ to classify it.

### Why does the update rule work?

A single perceptron update is not guaranteed to classify the point correctly immediately after the update. But it moves the decision boundary in a direction that makes the current example more likely to be classified correctly.

<p align="center">
  <img src = "../../../assets/img/perceptron/update-rule.png" alt="update-rule-geometry">
</p>

While adjusting the hyperplane for one point, it is possible that some previously correct points may become misclassified. However, we can show that if we keep following the update rule, the perceptron will find a separating hyperplane.

### Proof that the Perceptron will converge

The goal of this proof is to show that if the points are linearly separable, the Perceptron will find a separable hyperplane.

<p align="center">
  <img src = "../../../assets/img/perceptron/perceptron-proof1.jpeg" alt="perceptron algorithm">
  <br>
  <img src = "../../../assets/img/perceptron/perceptron-proof2.jpeg" alt="perceptron algorithm">
  <br>
  <img src = "../../../assets/img/perceptron/perceptron-proof3.jpeg" alt="perceptron algorithm">
</p>

## Usage

```
g++ perceptron.cpp ../../../utils/csv.cpp ../../../utils/metrics.cpp
```

## Results & Demo

We use perceptron for the task of gender classification using first names. To do this, we first construct features from the name using:

1) `26` unigrams
2) `26*26` bigrams
3) `26*26*26` trigrams
4) `1` feature indicating whether the name ends with a vowel.
5) `1` bias term

For Indian names, some of typical female names end in vowels. We therefore include a binary feature indicating whether the last character is a vowel. This is an example of mixing knowledge with data. However, this is very specific to our dataset and may not generalize to names from other regions. We validate the importance of these features using the validation set.

| Features                     | Validation Accuracy |
| ---------------------------- | ------------------: |
| All features                 |           **86.8%** |
| Without trigrams             |               76.7% |
| Without bigrams              |               83.7% |
| Without unigrams             |               85.2% |
| Without vowel-ending feature |               79.0% |

We observe that all features are contributing and hence we use them all for test set. The results on the test set are:

```
Accuracy: 0.868217
Precision: 0.847458
Recall: 0.862069
F1: 0.854701
```

We can now run the model on a few names that were not present in the dataset:

```
I am sure Asin is a boy.    # This is wrong
I am sure Raavan is a boy.
I am sure Mandodari is a girl.
I am sure Prabhas is a boy.
I am sure Zooni is a girl.
I am sure Chandanbala is a girl.
```