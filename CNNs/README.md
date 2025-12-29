# Convolutional Neural Networks

## Alexnet Architecture

<p align="center">
  <img src = "../assets/img/alexnet_architecture.png" alt="AlexNet">
</p>

## Plots

<p align="center">
  <img src = "../assets/img/alexnet_loss_curve.png" alt="AlexNet">
</p>

<p align="center">
  <img src = "../assets/img/alexnet_accuracy_curve.png" alt="AlexNet">
</p>

One observation that we can make is how the accuracy suddenly shoots up when we decrease learning rate. In SGD, we usually start with a high learning rate and slowly reduce it. This is like the exploration phase (high learning rate), and then exploitation phase (low learning rate).

## Results

The baseline classification accuracy for Imagenet 1K classification is `0.1%`. The AlexNet model achieves validation accuracy of `56%` after training for 80 epochs. We achieve slightly less accuracy than the original paper (`~63%`). Link to model checkpoint can be found <a href="https://iiithydresearch-my.sharepoint.com/:u:/g/personal/prit_kanadiya_research_iiit_ac_in/IQB7fEo9XRpGRrlg-ke4o4eUATMoGL8C1_22mJixxX-zmVc?e=shXKZr">here</a>.