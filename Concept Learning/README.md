# Concept Learning

## Introduction to Concept Learning

Imagine we want to learn to identify birds from a group of animals. We can start by looking at different birds and try to understand the features which make them different from other animals. We might look at a sparrow and derive that if the animal has wings, feathers, brown colour, a short tail etc. then it is a bird. However by looking at other birds, we might realize that colour has nothing to do with a bird, and we can come to generalized idea that if it has feathers or wings, it is a bird.

Here, bird becomes a Concept, which we want to learn. Concept Learning was the idea a boolean function defined over a set of all possible animal whicht returns true only if a given object is a member of a bird. The problem of inducing general functions from specific training examples is central to concept learning.

Formally, in the context of machine learning, we want to learn the function ```f(x)``` or hypothesis ```h```  within a hypothesis space ```H```, which can provide the most accurate approximation of the concept.

## Find-S Algorithm & Candidate Elimination Algorithm




















<img src = "../assets/birds-concept.png" alt= "Birds Concept">

## Introduction to Hypothesis Space & Concept Learning

Formally, in the context of machine learning, the function ```f(x)``` represents a hypothesis ```h```  within a hypothesis space ```H```. For instance, if we select a decision tree as our function ```f(x)```, then the hypothesis space ```H``` would encompass the set of all possible decision trees. The objective is to find a hypothesis ```h``` that serves as the most accurate approximation of the true function ```f```.

A concept is a well-defined collection of objects. Concept learning describes the process by which experience allows us to partition objects in the world into classes for the purpose of generalization. In simple words, Concept learning is learning how to group similar things together. We can say that a concept is a boolean function defined over a set of all possible objects and that it returns true only if a given object is a member of the concept. Otherwise, it returns false.

The problem of inducing general functions from specific training examples is central
to learning. For example, we see different species of birds such as sparrows, robins, and pigeons. Even though they look different, we notice they all have something in common. So, we create a concept in our mind called "birds" to group them together. Gradually, by looking at positive and negative examples, we move from a specific example of sparrow is a bird to general concept, that maybe if it has wings, it is a bird.

Consider the concept of a bird, where we want to infer whether an animal is a bird based on certain features. Let's define our features as follows:

* hasEyes: whether the animal has eyes (Boolean)
* hasWings: whether the animal has wings (Boolean)
* canFly: whether the animal can fly (Boolean)

The most specific hypothesis can be ```(None, None, None)``` which implies no animal is a bird. This implies that none of the animal can be a bird. (?, ?, ?) represents the most general definition of a bird, where any animal could potentially be classified as a bird regardless of its features. This implies that any combination of features could classify an animal as a bird. This hypothesis essentially suggests that none of the specified features are necessary for an animal to be considered a bird.

These hypotheses illustrate the extreme ends of the spectrum of possible classifications for the concept of a bird. The true nature of the concept might lie somewhere in between these two extremes, for example, (?, True, True) can be a possible hypothesis for the concept of bird.

## FIND-S Algorithm

### Assumptions

* Complete and Accurate Hypothesis Space: The algorithm assumes that the hypothesis space (H), which represents all possible descriptions of the target concept, is complete and contains a correct hypothesis. This means there exists at least one hypothesis in H that perfectly captures the true definition of the concept you are learning.

* Sufficient Positive Examples: The algorithm relies solely on positive examples to refine its hypothesis. It assumes that these positive examples are sufficient and informative enough to guide the learning process towards the correct concept. This means that positive examples should cover all aspects of the target concept and not be misleading or biased.

* Error-free Training Data: FIND-S assumes that the training examples are free from errors. This includes both positive examples (correctly classified instances) and negative examples (incorrectly classified instances). Any noise or mislabeled data in the training set can lead the algorithm to learn incorrect concepts.

### Algorithm

Find-S starts very specific and gets broader. 

### Intuition behind FIND-S

Imagine a detective trying to identify a specific type of criminal.

According to FIND-S Algorithm,

* we start with a very precise description (For example, "tall, blonde woman wearing a red jacket").
* As we interview witnesses, we adjust wer description to include everyone who matches the criminals they saw, even if that makes the description less specific. (For example, we might conclude that the criminal was atleast a tall woman by talking to all witnesses).
* we ignore people who didn't see the criminal at all.

Thus, we try to get a general view of the criminal by considering all the positive witnesses. This is first approach to understand who the criminal can be.

FIND-S provides a basic framework for concept learning through positive examples, but its limitations restrict its practical applicability in real-world scenarios. Other algorithms offer significant advantages in terms of robustness, adaptability, and effectiveness for various learning tasks. As a result, FIND-S hasn't received the same level of continued research and development as other techniques.