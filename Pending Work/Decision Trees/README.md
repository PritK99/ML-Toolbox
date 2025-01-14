# Decision Trees

**WIP**

## Algorithm

## Impurity Functions

### Gini Index

### Entropy

Entropy for a subset `S`, where proportion of positive and negative examples are given by `p(+)` and `p(-)` respectively is given by,

<img src="../assets/img/decision-tree-entropy.png" alt="entropy">

One way to look at entropy is that it represents the total number of bits required in order to express the class of a random instance of subset `S`. In other words, if we were to pick a random sample from `S`, what is the total number os bits required to express its class.

Consider a pure subset which consists only of positive elements. In this case, entropy = H(S) = 0. Thus, we dont require any bits to determine the class of the instance. We can be sure its a positive instance becuase the subset is positive.

Now, consider an impure subset such that there are 3 positives and 3 negatives. Here, we get `H(S) = 1`. This means that we require one full bit to determine the class of instance.

All this says is how uncertain the subset S is.

An explaination for this intuition can be found <a href="https://www.youtube.com/watch?v=tJmhT3oLXCU">here</a>