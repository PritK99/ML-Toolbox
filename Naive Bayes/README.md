# Naive Bayes

```
Every true genius is bound to be naive.
```

## Introduction

Naive Bayes is a generative learning model used for classification, distinct from discriminative models like perceptrons. Unlike discriminative models that focus on learning the decision boundary separating classes, Naive Bayes aims to understand the underlying distribution of data within each class. In simpler terms, it doesn't directly model the probability of a class (Y) given the data (X) which is ```(P(Y | X))```. Instead, it focuses on the probability of the data (X) given a specific class (Y) which is ```(P(X | Y))```. This essentially represents the likelihood of encountering a particular data point considering its class membership.

During prediction, when we need to determine P(Y | X), Naive Bayes employs Bayes theorem to arrive at the desired probability. This theorem provides a powerful tool for calculating conditional probabilities:

```P(Y | X) = (P(X | Y) * P(Y)) / P(X)```

Here, each term holds a specific meaning:

```P(Y)```: This represents the prior probability of class Y occurring in the dataset. It reflects the inherent bias towards a particular class if it exists.
```P(X)```: This denotes the prior probability of encountering data point X itself, independent of any class.
```P(X | Y)```: This term, known as the likelihood, is crucial. It represents the probability of observing data point X given that it truly belongs to class Y. This value is estimated using the training data.

However, a key challenge arises when dealing with high-dimensional data. Data point X is often a multi-dimensional vector, meaning it consists of numerous features. For instance, with 100 dimensions, X would be a vector of size 100, potentially containing specific values like "Red" and "Round". The problem lies in finding an exact match for this instance within the training data. As the number of dimensions increases, the probability of encountering an identical data point rapidly diminishes.

Imagine a single dimension with 100 possible values. Randomly picking a specific value has a probability of 0.01. Now, consider two such dimensions. The probability of finding a specific value in each dimension simultaneously drops to a mere 0.0001. This highlights the exponential decrease in probability with increasing dimensionality.

To address this challenge, Naive Bayes makes a simplifying assumption: conditional independence. This assumption, which is why the model is called "naive," posits that given a class label, the presence or absence of one feature is independent of the presence or absence of any other feature. In simpler terms, it assumes that the features provide independent clues about the class membership.

# Intuition

Naive Bayes makes a daring assumption: it believes that each feature, like color or size, doesn't rely much on the others. Surprisingly, this bold move doesn't hurt its performance much. That's why Naive Bayes can still give pretty good guesses, even though it's quite simple.

This bold move simplifies the math to:

```
P(X|Y) = [P(x1|Y) * P(x2|Y) * ... * P(xn|Y)] * P(Y) / P(X)
```

Calculating P(Y) is easy - just count how many examples you have for each class and make sure they all add up to one. As for P(X), it's like a constant that doesn't really change our guess. The big change comes when we break down ```P(x1|Y) * P(x2|Y) * ... * P(xn|Y)``` instead of directly dealing with ```P(X|Y)```. It's like shifting from searching in many dimensions to just one, which is much easier.

In a perfect world where all features are totally unrelated, Naive Bayes would be super accurate. But in reality, if features are connected, it might make errors. That's why Naive Bayes works best when features don't depend too much on each other.

<img src = "../assets/naive-bayes-intution.jpeg" alt="naive-bayes-intution">

## Algorithm