# KD & Ball Trees

## Table of Contents

- [KD & Ball Trees](#KD--Ball-Trees)
  - [Introduction](#introduction)
  - [Assumptions](#assumptions)
  - [Algorithm](#algorithm)
  - [Curse of Dimensionality](#curse-of-dimensionality)
  - [Usage](#usage)
  - [Results & Demo](#results--demo)

## Introduction

KD Trees and Ball Trees are data structures over KNN to make it faster. Both of them resursively partion data spatially.

## Assumptions

Both follow the same assumptions as KNN. LIke KNN, both suffer from curse of dimensionality. WHile KNN and Ball trees work for low intrincis dimensionllaity ike images, KD tree does not.

## Usage

To compile the code, run the following command:

```
g++ knn_fast.cpp ../../../utils/csv.cpp ../../../utils/distances.cpp ../../../utils/metrics.cpp kd.cpp
```