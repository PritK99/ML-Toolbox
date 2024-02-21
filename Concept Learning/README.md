# Concept Learning

<img src = "../assets/birds-concept.png" alt= "Birds Concept">

## Introduction to Hypothesis Space & Concept Learning

Formally, in the context of machine learning, the function ```f(x)``` represents a hypothesis ```h```  within a hypothesis space ```H```. For instance, if we select a decision tree as our function ```f(x)```, then the hypothesis space ```H``` would encompass the set of all possible decision trees. The objective is to find a hypothesis ```h``` that serves as the most accurate approximation of the true function ```f```.

Concept learning describes the process by which experience allows us to partition objects in the world into classes for the purpose of generalization. In simple words, Concept learning is learning how to group similar things together. Each concept can be thought of as a boolean-valued function defined over some larger set.

The problem of inducing general functions from specific training examples is central
to learning. For example, we see different species of birds such as sparrows, robins, and pigeons. Even though they look different, we notice they all have something in common. So, we create a concept in our mind called "birds" to group them together. Gradually, by looking at positive and negative examples, we move from a specific example of sparrow is a bird to general concept, that maybe if it has wings, it is a bird.

Thus, Concept learning is inferring a boolean-valued function from training examples of
its input and output.

Consider the concept of a bird, where we want to infer whether an animal is a bird based on certain features. Let's define our features as follows:

* hasEyes: whether the animal has eyes (Boolean)
* hasWings: whether the animal has wings (Boolean)
* canFly: whether the animal can fly (Boolean)

The most specific hypothesis can be ```(None, None, None)``` which implies no animal is a bird. This implies that none of the animal can be a bird. (?, ?, ?) represents the most general definition of a bird, where any animal could potentially be classified as a bird regardless of its features. This implies that any combination of features could classify an animal as a bird. This hypothesis essentially suggests that none of the specified features are necessary for an animal to be considered a bird.

These hypotheses illustrate the extreme ends of the spectrum of possible classifications for the concept of a bird. The true nature of the concept might lie somewhere in between these two extremes, for example, (?, True, True) can be a possible hypothesis for the concept of bird.

## FIND-S Algorithm

FIND-S Algorithm tries to learn the best hypothesis 

Imagine learning the concept of what an ideal day might be for picnic, 