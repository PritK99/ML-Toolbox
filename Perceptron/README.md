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

## Proof that Perceptron will always converge

If the points are linearly separable (can be perfectly separated by a line or hyperplane), the Perceptron will converge to a solution (though not necessarily the best one). But if they're not separable, it will loop forever.

<a href="https://www.youtube.com/watch?v=fHDouTKwfXw">Click Here</a> for proof that the Perceptron will always converge if the data fits our assumption.

## Results

Perceptron algorithm for Gender Classification failed to converge due to 2 missclassifications. Using First Names, the model achieved an accuracy of 85% on the test data. This observation highlights the effectiveness of the perceptron in higher dimensions, where it can better separate classes. We tried to increase the feature space by using quadgrams, but the algorithm failed to converge for that case as well.