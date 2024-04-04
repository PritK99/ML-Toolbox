# Logistic Regression

## Introduction

Logistic regression is the discriminative counterpart of Naive Bayes. Hence it assumes a parametric form over P(Y | X) and finds parameters for this directly. This is different from Naive Bayes which learns P(X | Y) and P(Y). 

Consider the example of classification between cats and dogs. A generative model like naive bayes tries to understand what features makes an animal dog. Thus it tries to draw an image of what a dog might be. However discriminative model on the other hand tries to find the features which can distinguish a dog from cat. To put it simply, Generative model tries to understand what makes an animal a dog or a cat, while Discriminative model focuses on pinpointing the key features that tell them apart.

## Algorithm

Logistic regression assumes the following parametric form over P(Y | X),



## Naive Bayes vs Logistic Regression

Logistic regression offers several advantages over Naive Bayes. One key advantage is that Naive Bayes makes overly strong assumptions about the independence of features. Imagine two features that are highly correlated, like having the same feature counted twice. Naive Bayes treats each copy of the feature as separate, which can lead to overestimating the evidence. However, logistic regression handles correlated features better. If two features are perfectly correlated, logistic regression splits the weight between them, leading to more accurate probabilities when there are many correlated features.

Logistic regression generally performs better on larger datasets or documents and is often the default choice. Although Naive Bayes may not provide the most accurate probabilities due to its assumptions, it often still makes the correct classification decision. Additionally, Naive Bayes can excel on small datasets or short documents and is easy to implement and train quickly since there's no optimization step involved. So, depending on the situation, Naive Bayes remains a reasonable choice for classification tasks.