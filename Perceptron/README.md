# Perceptron

<img src="../assets/img/perceptron-visualization.png" alt="perceptron-visualization">

## Assumptions

Perceptrons are used for binary classification. They assume that there's at least one straight line (or plane, in higher dimensions) that can perfectly separate the two classes. While this isn't always true in low dimensions, it tends to hold in higher dimensions because points are more spread out, making separation easier. So, for low dimensions, other algorithms like KNN are preferred, while for higher dimensions, perceptrons are useful.

## Algorithm

In a perceptron, we define a hyperplane using a weight vector `w` (normal to the hyperplane) and a bias `b`. We can combine these into one vector `W` = `[w, b]` by extending our feature space into one higher dimension. Essentially, we're absorbing the bias term as one dimension and lifting our points into a higher dimension.

A hyperplane is like a flat sheet, one dimension lower than the feature space. For instance, in 2D, it's a line. Instead of characterizing it with both `w` and `b`, we extend it to 3D and draw a plane passing through the origin, removing the need for an extra bias term.

<img src = "../assets/img/perceptron.jpeg" alt="Perceptron Algorithm">

During inference, we look at the direction of the point relative to the hyperplane using `W.T @ x` and classify based on the sign of the result. If `W.T @ x > 0`, it's in the positive class, and vice versa.

However, if the data isn't linearly separable, the algorithm will keep trying indefinitely. To avoid this, we can set a limit on the number of iterations.

### Geometric Intuition

Adding `yx` of incorrectly classified example to weights `W` fixes the issue of misclassifciation because we add (if `Y = +1`) or subtract (if `Y = -1`) the data point vector to normal vector which fixes the issue.

<img src="../assets/img/perceptron-working.jpeg" alt="perceptron-working">

## Proof that Perceptron will always converge

If the points are linearly separable (can be perfectly separated by a line or hyperplane), the Perceptron will converge to a solution (though not necessarily the best one). But if they're not separable, it will loop forever.

<img src="../assets/img/perceptron-proof1.jpeg" alt="perceptron proof">

<img src="../assets/img/perceptron-proof2.jpeg" alt="perceptron proof">

<img src="../assets/img/perceptron-proof3.jpeg" alt="perceptron proof">

<a href="https://www.youtube.com/watch?v=fHDouTKwfXw">Click Here</a> for video explanation of proof that the Perceptron will always converge if the data fits our assumption.

## Results

The Perceptron algorithm for Gender Classification converges in `52` steps when we consider features such as the last character, bigrams, and trigrams from the name. It achieves an accuracy of `85.38%` on the test data. However, when we simplify the features to only include the last character and bigrams, the algorithm struggles to converge. This indicates that the Perceptron excels in higher dimensions, where it can more effectively distinguish between classes.