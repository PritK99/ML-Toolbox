# ML-Toolbox

<p align="center">
    <img src="assets/img/logo.png" alt="logo">
</p>

## Table of Contents

- [Project](#ml-toolbox)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Introduction to Machine Learning](#introduction-to-machine-learning)
    - [Traditional Programming vs Machine Learning](#traditional-programming-vs-machine-learning)
    - [Core Idea Behind Machine Learning](#core-idea-behind-machine-learning)
  - [Introduction to ML-Toolbox](#introduction-to-ml-toolbox)
  - [File Structure](#file-structure)
  - [References](#references)
  - [License](#license)

## About

Each machine learning algorithm is based on certain assumptions about the data, has an underlying theoretical foundation, and comes with its own set of advantages and disadvantages. In this repository, I aim to build intuition, implement algorithms from scratch, and derive the theoretical foundations and proofs behind them. By understanding both the theory behind an algorithm and the characteristics of the data, we can make more informed choices, improve performance, and achieve better results.

## Introduction to Machine Learning

### Traditional Programming vs Machine Learning

Traditional programming is based on the idea of writing a program, providing it with input, and receiving an output. This approach works well for tasks where the rules can be clearly defined. For example, classifying a number as odd or even can be easily handled by a simple if-else program.

<img src = "assets/img/Traditional CS.png" alt="ML">

However, for problems where the rules cannot be clearly defined, we turn to machine learning to automatically generate these rules from data. Consider the problem of classifying an image as a cat or a dog. Writing a rule-based program for this would be extremely difficult. Machine Learning is the process in which we provide a computer with data and corresponding outputs (labels), and it learns a function to map inputs to outputs. This phase is called training. Once trained, we can use the model on new data to make predictions, much like traditional programming. This phase is known as inference.

<img src = "assets/img/ML.png" alt="ML">

### Core Idea behind Machine Learning

Machine Learning is a subset of Artificial Intelligence (AI). While AI broadly aims to mimic human intelligence and decision-making, ML focuses on using statistics to uncover patterns in data. For example, in games like chess, traditional AI might use algorithms like minimax, mimicking strategic human thinking. In contrast, ML methods such as linear regression use statistical techniques focusing on data-driven pattern recognition rather than explicitly imitating human strategies.

<img src="assets/img/ml-idea.jpg" alt="ml-idea">

he main goal of ML is to discover the underlying (but unknown) process that generates the observed data. We often model this as a probability distribution `P(x, y)`, which captures the relationship between inputs (`x`) and outputs (`y`) in the real world.

## Introduction to ML-Toolbox

The ML-Toolbox is like a toolkit of various machine learning methods, each offering its own approach to modeling a function `f(x)`. The key is choosing the right tool for the task at hand, depending on the problem we're trying to solve. While neural networks are widely used, they’re just one tool in the box which produce outputs in the form of weights and biases.

The core idea behind the ML-Toolbox is to develop an understanding of a broad range of algorithms, including Decision Trees, Neural Networks, Support Vector Machines, Random Forests, and K-Nearest Neighbors. The goal isn’t to declare one method as the best, but rather to understand the strengths and weaknesses of each. It's like knowing when to use a screwdriver instead of a hammer.

## File Structure

```
ML-Toolbox/
 ┣ 📂assets/                                 # Supporting resources
 ┃ ┣ 📂data/                                 # Datasets used in experiments and examples
 ┃ ┃ ┣ 📄articles.csv
 ┃ ┃ ┣ 📄gender.csv
 ┃ ┃ ┣ 📄modified_mumbai_house_price.csv
 ┃ ┃ ┣ 📄mumbai_house_price.csv
 ┃ ┃ ┣ 📄student_marksheet.csv
 ┃ ┃ ┣ 📄titanic.csv
 ┃ ┃ ┣ 📄un_voting.csv 
 ┃ ┣ 📂img/                                  # Images used in documentation or notebooks
 ┃ ┣ 📂scripts/                              # Utility or preprocessing scripts
 ┃ ┣ 📂notes/                                # Theoretical notes and derivations, 
 ┣ 📂Concept Learning/                       # Implementations and theory behind concept learning
 ┣ 📂K Nearest Neighbors/
 ┣ 📂Perceptron/
 ┣ 📂Naive Bayes/
 ┣ 📂Logistic Regression/
 ┣ 📂Linear Regression/
 ┣ 📂Support Vector Machine/
 ┣ 📂Kernels/                                # Kernelized versions of various algorithms
 ┃ ┣ 📂Perceptron/
 ┃ ┣ 📂Linear Regression/
 ┃ ┣ 📂Support Vector Machine/
 ┣ 📂Decision Trees/
 ┣ 📂Neural Networks/
 ┣ 📂K Means Clustering/
 ┣ 📄README.md                               # Project overview
```

## References

* A big thanks to Prof. Kilian Weinberger for the Cornell CS4780 course, <a href="https://www.youtube.com/playlist?list=PLl8OlHZGYOQ7bkVbuRthEsaLr7bONzbXS">Machine Learning for Intelligent Systems</a>. Majority of the content in this repository is inspired by the lectures.
* MIT 6.036 <a href="https://www.youtube.com/playlist?list=PLxC_ffO4q_rW0bqQB80_vcQB09HOA3ClV">Machine Learning</a> by Prof. Tamara Broderick.
* Bias Variance Tradeoff by <a href="https://ocw.mit.edu/courses/15-097-prediction-machine-learning-and-statistics-spring-2012/dec694eb34799f6bea2e91b1c06551a0_MIT15_097S12_lec04.pdf" target="_blank">MIT OpenCourseware</a> and <a href="https://nlp.stanford.edu/IR-book/html/htmledition/the-bias-variance-tradeoff-1.html" target="_blank">The Stanford NLP Group</a>.
* Additional resources to understand <a href="https://ml-course.github.io/master/notebooks/03%20-%20Kernelization.html">kernelizations</a>.
* <a href="http://neuralnetworksanddeeplearning.com/index.html">Neural Networks and Deep Learning</a> Online Book by Michael Nielsen.
* <a href="https://www.visiondummy.com/2014/04/curse-dimensionality-affect-classification/">Blog</a> on the Curse of Dimensionality.
* <a href="https://www.kaggle.com/">Kaggle</a> for providing several datasets used in this repository.

## License
[MIT License](https://opensource.org/licenses/MIT)