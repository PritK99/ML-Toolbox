# KD & Ball Trees

## Table of Contents

- [KD & Ball Trees](#KD--Ball-Trees)
  - [Introduction](#introduction)
  - [Assumptions](#assumptions)
  - [Algorithm](#algorithm)
  - [Usage](#usage)
  - [Results & Demo](#results--demo)

## Introduction

KD Trees and Ball Trees are data structures that can make KNN faster. Both recursively partition the data into smaller regions so that we do not have to compare a query point with every point in the dataset.

## Assumptions

Both methods follow the same assumptions as KNN, and hence also suffer from the curse of dimensionality.

Like KNN, Ball Trees can work well when the data has a low intrinsic dimensionality, even if the number of features is relatively large. KD Trees don't work with higher dimensions, even if the intrinsic dimensionality is low.

## Algorithm

## Usage

To use the KD Tree, uncomment line `26` in `knn_fast.cpp`, which declares the KD Tree namespace, and comment out line `27`, which declares the Ball Tree namespace. To use the Ball Tree, do the opposite, i.e. uncomment line `27` and comment out line `26`. The function signatures are the same for both implementations, so the rest of the code does not need to change.

To compile the code, run the following command:

```
g++ knn_fast.cpp ../../../utils/csv.cpp ../../../utils/distances.cpp ../../../utils/metrics.cpp kd.cpp ball.cpp
```

## Results & Demo

One way to validate our implementation is to compare its predictions with vanilla KNN. The predictions should be the same because KD Trees and Ball Trees are simply data structures on top of KNN. They do not change the underlying algorithm

Similar to KNN, we split the data into an 80:10:10 train-validation-test split and found `K = 33` with Euclidean distance to be optimal on the validation set. Using the same configuration, we obtained the following results on the test set for both algorithms:

```
MAE: 0.523629 
RMSE: 0.685546 
R2: 0.5643
```

The KD Tree implementation took `8664 ms`, compared to `12065 ms` for vanilla KNN. This gives a speedup of approximately `28%`, while producing the same predictions as vanilla KNN.

Sadly, the Ball Tree implementation took `17520 ms`, which is about `45%` slower than vanilla KNN. This could be due to a suboptimal implementation. 

We also ran the model on the same essay used for KNN. It achieved a score of 1.82.

`This is a test essay. Lets see where it goes! This is a poem from the family guy. Oh, squiggly line in my eye fluid, I see you lurking there on the periphery of my vision. But when I try to look at you, you scurry away. Are you shy, squiggly line? Why only when I ignore you, do you return to the center of my eye? Oh, squiggly line, it's alright, you are forgiven.`