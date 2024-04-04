# ML-Toolbox

## Introduction to Machine Learning

Traditional Programming is based on the idea of writing a program, giving it an input and getting an output. This works well for all the tasks where the rules are clearly defined. Consider the problem of classifying a number as odd or even. This can be done by a simple if-else program.

However for problems where the rules are not clearly defined, we need to use Machine Learning. Consider the problem of classifying an image as cat or dog. Writing a program for this would be very difficult. Machine Learning is the idea where we provide the computer with data and corresponding outputs and get the program. This phase is called training. Now we use these program, along with new data like traditional programming to get an output. This phase is called Testing.

<img src="assets/intro-comparison.png" alt="Comparison between traditional programming and machine learning">

The core idea in solving any task in machine learning is to find a distribution or a function ```y = f(x)```, where ```y``` represents the output and ```x``` represents the input. In traditional programming, you typically define this function manually. However, in machine learning, you delegate this task to the computer, allowing it to learn the function from data. The computer examines the data provided and attempts to derive a function that closely approximates the true underlying function present in the data.

Nevertheless, this approach increases the reliance on the data used to train the model. The probability distribution of the output is heavily influenced by the characteristics of the data employed during the training process. Consequently, a model trained on data from a specific context or environment may struggle to generalize well to different contexts or environments. For instance, a self-driving car trained on the streets of Europe might not perform optimally when deployed on the streets of India due to variations in traffic patterns, road conditions, and driving norms. 

## Intorduction to ML-Toolbox

The main idea behind ML-Toolbox is to understand the different forms of ```f(x)``` that can be generated. These forms may include functions, decision trees, neural networks, support vector machines, random forests, etc. Thus, ML-Toolbox is a collection of various ML algorithms, each providing a specific form of ```f(x)```. The choice of which form of ```f(x)``` to use depends on the problem being addressed and can be considered a type of hyperparameter. Neural Networks represent just one type of ```f(x)```, with the program outputted in the form of weights and biases. This field of deep learning hence is considered as subset of Machine Learning.

Some of the popular choices for f(x) are:

* Descision Trees
* Neural Networks
* Support Vector Machines
* Random Forests
* K-Nearest Neighbors

A loss function allows us to measure the performance of the ```f(x)```. Loss functions are always non-negative and essentially indicate how far the ```f(x)``` is from the true function. However, the loss function serves not only for evaluation but also for training, as it guides the optimization and training process. For example, in neural networks, the loss function is used to calculate the gradients of the weights and biases, which are then used to update the weights and biases.

The choice of loss function is also very important. It can not be said for example that square loss is always better than absolute loss. Absolute loss is preferred over square loss in scenarios where outliers have a significant impact on the model's performance. Since absolute loss is less sensitive to outliers compared to square loss, it can provide more robust performance in such cases. For instance, in regression tasks where outliers are present and need to be handled with care, absolute loss may be a better choice as it penalizes large errors linearly, while square loss penalizes them quadratically, making it more sensitive to extreme values.

## Finding the Optimal Function

Formally, in the context of machine learning, the function ```f(x)``` represents a hypothesis ```h```  within a hypothesis space ```H```. For instance, if we select a decision tree as our type of function ```f(x)```, then the hypothesis space ```H``` would encompass the set of all possible decision trees. The objective is to find a hypothesis ```h``` that serves as the most accurate approximation of the true function ```f```.

The primary aim is to find the probability distribution or function ```f(x)```. However, directly uncovering this function is often unattainable. Therefore, we resort to approximating it using available data. A subset of this data is employed to approximate the function, while another subset is reserved to evaluate the proximity of this approximation. The former subset is known as the training data, while the latter is referred to as the test data. The split between the train and test data is pivotal and varies depending on the problem at hand. For instance, if our data exhibits a temporal aspect, we should partition it based on time. Conversely, if the data is independently and identically distributed, a random split or shuffling suffices.

In addition to the train and test datasets, we also utilize a validation dataset. The rationale behind incorporating a validation dataset is to adhere to the principle of only utilizing a given test set once. Iteratively refining a model based on feedback from the test set is discouraged, as it can lead to overfitting. The validation set serves the purpose of fine-tuning hyperparameters or even altering the type of function employed. However, once the model is evaluated using the test set, the reported accuracy should reflect the final performance. This practice safeguards against overfitting to the test dataset, ensuring that the estimated probability remains as close as possible to the true probability distribution.

The choice of hypothesis space ```H``` heavily relies on the assumptions made about the data. A common mistake is to favor a particular type of function, such as Neural Networks, without considering the characteristics of the data. Each machine learning algorithm operates under different assumptions about the data, and these assumptions influence the probability it generates. Therefore, it's essential to select the algorithm that best aligns with the assumptions inherent in the data. By understanding these underlying assumptions and choosing the appropriate algorithm accordingly, we increase the likelihood of constructing a model that accurately captures the data's patterns and produces reliable predictions.


Pending:

FIND S README
LINEAR README
LOGISTIC README
NAIVE BAYES README
KNN README AND REORGANIZATION
DECISION TREE CODE AND README