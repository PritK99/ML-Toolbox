# Perceptron

<img src="../assets/img/perceptron-visualization.png" alt="perceptron-visualization">

## Assumptions

Perceptrons are used for classification, and they make a core assumption that there exists a hyperplane that can perfectly separate the data. While this isn't always true in low dimensions, it tends to hold in higher dimensions because points are more spread out due to the curse of dimensionality, making separation easier. So, for low dimensions, other algorithms like KNN are preferred, while for higher dimensions, perceptrons are useful.

## Algorithm

In a perceptron, we define a hyperplane using a weight vector `w` (normal to the hyperplane) and a bias `b`. We can combine these into one vector `W` = `[w, b]` by extending our feature space into one higher dimension. A hyperplane is a subspace that is one dimension lower than the feature space. For instance, in 2D, it's a line. Instead of characterizing it with both `w` and `b`, we extend it to 3D and draw a plane passing through the origin, removing the need for an extra bias term.

<img src = "../assets/img/perceptron.jpeg" alt="Perceptron Algorithm">

During inference, we look at the direction of the point relative to the hyperplane using `Wx` and classify based on the sign of the result. If `Wx > 0`, it's in the positive class, and vice versa.

However, if the data isn't linearly separable, the algorithm will keep trying indefinitely. To avoid this, we can set a limit on the number of iterations.

### Geometric Intuition

Adding `yx` of incorrectly classified example to weights `W` pushes the hyperplane to accommodate that missclassified point. While it is not guaranteed that the point will be fixed immediately, the perceptron will slowly get that point right. This is because we add (if `Y = +1`) or subtract (if `Y = -1`) the data point vector from the normal vector, which fixes the issue.

<img src="../assets/img/perceptron-working.jpeg" alt="perceptron-working">

However, this intuition is not enough to know that perceptron will converge for all data point. While accommodating the misclassified point, it may make new mistakes for orignally classified points. Hence we need to prove that the perceptron will converge given our assumption holdes true.

## Proof that the Perceptron will always converge

If the points are linearly separable (can be perfectly separated by a line or hyperplane), the Perceptron will converge to a solution (though not necessarily the best one). But if they're not separable, it will loop forever.

<img src="../assets/img/perceptron-proof1.jpeg" alt="perceptron proof">

<img src="../assets/img/perceptron-proof2.jpeg" alt="perceptron proof">

<img src="../assets/img/perceptron-proof3.jpeg" alt="perceptron proof">


## Results

The Perceptron algorithm for Gender Classification converges in `98` steps when we consider feature vector made up of all unigrams, all bigrams, and all trigrams in name and a binary feature indicating whether the last character is a vowel i.e. vector of size `(18280, 1)`. It achieves an accuracy of `88.46%` on the test data, which is a good score compared to other algorithms such as KNN. 

Sample Predictions:

```
Indian Names:
I am sure Asin is a boy.    # Wrong
I am sure Anvay is a boy.
I am sure Samantha is a girl.
I am sure Raavan is a boy.
I am sure Mandodari is a girl.
I am sure Zooni is a girl.
I am sure Chandanbala is a girl.

Foreign Names:
I am sure Emma is a girl.
I am sure Jacob is a boy.
I am sure Carlos is a boy.
I am sure Hermoine is a girl.
I am sure Leonardo is a girl.    # Wrong
I am not sure if Meryl is a boy or a girl.    # Wrong
I am sure Obama is a girl.    # Wrong
```