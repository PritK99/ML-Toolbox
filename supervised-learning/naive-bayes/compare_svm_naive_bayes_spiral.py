"""
This script compares the decision boundaries of two classifiers — 
Support Vector Machine (SVM) with an RBF kernel and Gaussian Naive Bayes (NB) — 
on a spiral-shaped dataset.

The spiral dataset and code is adapted from:
"How to classify data which is spiral in shape?"
https://stats.stackexchange.com/questions/235600/how-to-classify-data-which-is-spiral-in-shape

The goal is to visualize how SVM (a nonlinear classifier) and Naive Bayes (a generative, 
feature-independent model) differ in handling complex nonlinear data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Load data
f = np.loadtxt("../assets/data/spiral.data")
x, y = f[:, :2], f[:, 2]

# Create classifiers
svm_clf = SVC(kernel='rbf', gamma=100)
nb_clf = GaussianNB()

# Fit models
svm_clf.fit(x, y)
nb_clf.fit(x, y)

# Create mesh grid
xmin, xmax = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
ymin, ymax = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01),
                     np.arange(ymin, ymax, 0.01))
xnew = np.c_[xx.ravel(), yy.ravel()]

# Predict
svm_pred = svm_clf.predict(xnew).reshape(xx.shape)
nb_pred = nb_clf.predict(xnew).reshape(xx.shape)

# Plot
plt.figure(figsize=(12, 5))
plt.set_cmap(plt.cm.Paired)

# SVM plot
plt.subplot(1, 2, 1)
plt.pcolormesh(xx, yy, svm_pred, shading='auto')
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k')
plt.title("SVM (RBF Kernel)")

# Naive Bayes plot
plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, nb_pred, shading='auto')
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k')
plt.title("Naive Bayes")

plt.tight_layout()
plt.show()
