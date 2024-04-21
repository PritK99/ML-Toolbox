# ML-Toolbox

## Introduction to Machine Learning

### Traditional Programming vs Machine Learning

Traditional Programming is based on the idea of writing a program, giving it an input and getting an output. This works well for all the tasks where the rules can be clearly defined. Consider the problem of classifying a number as odd or even. This can be done by a simple if-else program.

<img src = "assets/img/Traditional CS.png" alt="ML">

For problems where the rules can not be clearly defined, we use Machine Learning to generate these rules for us. Consider the problem of classifying an image as cat or dog. Writing a program for this would be very difficult. Machine Learning is the idea where we provide the computer with data and corresponding outputs and get the program. This phase is called training. Now we use these program, along with new data like traditional programming to get an output. This phase is called Testing.

<img src = "assets/img/ML.png" alt="ML">

### Core Idea behind Machine Learning

Machine Learning is a subset of Artificial Intelligence (AI). While AI aims to imitate human thinking, Machine Learning focuses on using statistics to uncover patterns in data. For instance, in games like chess, AI uses strategies like minimax, similar to how humans strategize, while Machine Learning methods such as Linear Regression aim to draw the best-fitting line through data points, relying on statistics and pattern recognition rather than mimicking human thought processes.

At the heart of machine learning is the quest to find a function `f(x)` that closely approximates the relationship between inputs and outputs in the real world. Unlike traditional programming, where functions are manually defined, machine learning algorithms learn from data to automatically derive the most suitable function or model for a given task.

## Introduction to ML-Toolbox

The ML-Toolbox is like a toolkit full of different machine learning methods, each offering its own form of`f(x)`. The trick is picking the right one for the job, which is kind of like choosing a setting on a tool – it depends on what we are trying to do. Neural networks are popular, but they're just one tool in the box, giving us outputs in the form of weights and biases.

The core concept behind the ML-Toolbox is to grasp the diverse range of algorithms capable of generating forms of `f(x)`. Some widely used algorithms include Decision Trees, Neural Networks, Support Vector Machines, Random Forests, and K-Nearest Neighbors. The goal isn't to say which method is the best. Instead, it's about knowing when each method works well and when it might struggle. It's like knowing when to use a screwdriver versus a hammer.

## File Structure
```
👨‍💻ML-Toolbox
 ┣ 📂assets                                   
 ┃ ┣ 📂data                                 // datasets  
 ┃ ┃ ┣ 📄titanic.csv
 ┃ ┃ ┣ 📄mumbai_house_price.csv
 ┃ ┃ ┣ 📄gender.csv
 ┃ ┣ 📂img                     
 ┣ 📂Concept Learning                       // Classification
 ┃ ┣ 📄titanic survival prediction.ipynb
 ┃ ┣ 📄README.md
 ┣ 📂K Nearest Neighbors                    // Classification & Regression
 ┃ ┣ 📄gender prediction.ipynb
 ┃ ┣ 📄house price prediction.ipynb
 ┃ ┣ 📄README.md
 ┣ 📂Perceptron                             // Classification
 ┃ ┣ 📄gender prediction.ipynb
 ┃ ┣ 📄README.md
 ┣ 📂Naive Bayes                            // Classification
 ┃ ┣ 📄gender prediction.ipynb
 ┃ ┣ 📄README.md
 ┣ 📂Logistic Regression                    // Classification
 ┃ ┣ 📄gender prediction.ipynb
 ┃ ┣ 📄README.md
 ┣ 📂Linear Regression                      // Regression
 ┃ ┣ 📄house price prediction.ipynb
 ┃ ┣ 📄README.md
 ┣ 📂Support Vector Machine                 // Classification
 ┃ ┣ 📄gender prediction.ipynb
 ┃ ┣ 📄README.md
 ┣ 📄README.md
``` 

## References

* `Bias Variance Tradeoff` by <a href="https://ocw.mit.edu/courses/15-097-prediction-machine-learning-and-statistics-spring-2012/dec694eb34799f6bea2e91b1c06551a0_MIT15_097S12_lec04.pdf">MIT OpenCourseware</a> and <a href="https://nlp.stanford.edu/IR-book/html/htmledition/the-bias-variance-tradeoff-1.html">The Stanford NLP Group</a>.
 
## License
[MIT License](https://opensource.org/licenses/MIT)