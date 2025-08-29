# K Nearest Neighbors (KNNs)

<img src="../assets/img/knn1.webp" alt="KNN">

## Introduction

K Nearest Neighbors (KNNs) is a supervised machine learning algorithm that can be used for both classification and regression tasks. 

## Assumptions

KNN operates under the assumption that instances that are close to each other in the feature space are likely to be similar. In other words, points that are similar to each other also tend to have similar target values. Thus the target value of a new instance is likely to be the same as its nearest neighbors.

However, this assumption may not always hold, especially in datasets where the relationship between features and target values is not strictly dependent on proximity. For instance, consider a dataset concerning customer preferences, where features such as age and income are considered. Customers with similar ages and incomes may still have vastly different preferences.

## Algorithm

The KNN Algorithm simple looks for the K nearest neighbors based on a distance metric, and takes a vote among them. For regression, it can be average of all values.

The KNN algorithm is only as good as its distance metric. The distance metric should be such that it captures the similarity between instances appropriately. 

For instance, Euclidean distance works well for classifying handwritten digits because it captures the geometric closeness of pixel intensities in image space. However, using Euclidean distance for comparing text documents isn't ideal, since text data is typically represented as high-dimensional sparse vectors (e.g., word counts or TF-IDF scores). In such cases, Cosine similarity is a better choice because it measures the angle between two vectors, focusing on their direction rather than their magnitude. This makes it more suitable for identifying how similar two texts are based on their content, regardless of document length or scale.

Some commonly used metrics are as follows:-

1. Minkowski Distance: This is a generalized distance metric that includes Manhattan (p=1), Euclidean (p=2), and Chebyshev (p=infinity) as special cases. It is defined as:

<img src="../assets/img/minkowski-distance.png" alt="Minkowski Distance">

The choice of distance metric depends on the amount of penalty one wants to assign to differences in each dimension. If p is lower, say 1, then the metric is less sensitive to outliers and treats each dimension equally. If p is higher, say infinity, then it is sensitive to outliers in any single dimension.

Consider analyzing user behavior on an e-learning platform, with features like number of quizzes taken, videos watched, time spent reading, and forum activity. These actions are independent and contribute additively to overall engagement. Manhattan distance is better here as it effectively captures similarity in such multi-faceted user behavior without letting one large difference dominate. Euclidean distance (p=2) would work better. Similarly, Chebyshev distance (p=infinity) can be used when one dimension is more important than others. For example, in medical diagnosis, a single severely abnormal symptom may be more significant than several marginally abnormal symptoms.

2. Cosine Similarity: Cosine similarity measures the cosine of the angle between two vectors. It is a measure of orientation and not magnitude. Cosine similarity is commonly used as a similarity metric for text data.

Once the similarity metric is defined, for any given test point, we look at its `k` nearest neighbors and take the majority vote of the classes of these neighbors for classification and average of the target values for regression. 

The choice of the parameter `k` (the number of neighbors) is crucial, as it impacts the model's sensitivity to noise and generalization ability. A smaller ```k``` may result in a model that is sensitive to noise, while a larger ```k``` may lead to a model that is too generalized. The optimal `k` is often determined through validation methods.

## Curse of Dimensionality in KNNs

<img src="../assets/img/curseofdimensionality.png" alt="curse-of-dimensionality">

KNNs are based on the assumption that data points close together in the feature space are more likely to belong to the same category. However, as the number of features increases, this assumption can break down due to the curse of dimensionality. In high-dimensional spaces with few data points (sparse data), identifying the true nearest neighbors becomes challenging.

One consequence of this challenge is that the nearest neighbor found by the algorithm might not truly be a neighbor in the meaningful sense. In reality, it could be far from the test point, appearing close only due to the sparseness of the data. Consequently, the core assumption of KNN that nearby points are similar becomes meaningless in such scenarios.

In these cases, algorithms like the perceptron may be more suitable for classification tasks. The perceptron, for instance, can handle higher dimensions more gracefully and is less affected by the curse of dimensionality.

However, it's essential to note that there are instances where datasets possess large dimensions but low intrinsic dimensionality. In such cases, KNN can still be effective. For example, images often have high dimensions but low intrinsic dimensionality, meaning that important information can be captured in fewer dimensions. Techniques like Principal Component Analysis (PCA) can help in reducing the dimensions while preserving most of the important information, making KNN applicable even in high-dimensional scenarios.

## Results

### Classification

Our goal was to build a system that could classify names as belonging to boys or girls. We had two options:

* Full Name as Text: This keeps the name as it is, "John" or "Sarah".
* Encoded Name: We converted the name into a vector of 702 numbers. This vector was made up of the last letter and bigrams (like "ia" or "th").

We used various distance metrics such as Manhattan distance, Euclidean Distance, Cosine Similarity, Hamming Distance to measure the distance between vectors, but none of these methods worked well. This might be because we didn't have enough data to make sense of such a complex representation. This might be due to curse of dimensionality.

Since that didn't work, we decided comparing the names directly as text. We used minimum edit distance as the distance metric. This calculates the minimum number of changes (insertions, deletions, or replacements) needed to turn one name into another. This method of using minimum edit distance as distance metric proved to be more effective and achieved an accuracy of `82.45%` on test data with `K=17`. 

However this method has a shortcoming. Consider the names 'Prit' and 'Priti'. Clearly, the vowel on the end changes the gender. However, both of these names differ by edit distance of 1. Thus, we design a new metric for Indian names, accounting for the fact that addition of a vowel in end changes gender. A special adjustment is made to edit distance when the edit distance is 1 and the names only differ by a vowel at the end (e.g., "Shrey" and "Shreya"). In such cases, based on domain knowledge, the labels of the training samples are swapped (using XOR on the label), which might represent handling specific domain nuances. This method provided a validation accuracy of `86.82%` and a test accuracy of `83.84%` for `K=17`. 

Further, by using weighted KNN woth `K=17`, we get a validation accuracy of `87.82%` and a test accuracy of `86.15%`.

### Regression

For the task of Mumbai House Price Prediction, average errors for KNN is `0.38 Cr`, but weighted KNN has a lower error of `0.30 Cr`. Here, we use the inverse distances as weights.

This is quite lower error compared to linear regression. This is maybe because of features such as latitude and longitude. In places like Mumbai, where location strongly affects prices, linear regression falls short because it can't handle non-linear relationships like those between prices and coordinates. Also, by converting nominal and ordinal features such as age, type, and status to appropriate numeric values, we were able to improve accuracy. We use absolute distance as a metric for finding the nearest neighbour. This absolute distance represents median value and hence is more effective than linear regression which predicts the mean. While predicting quantities such as property prices or average salary, outliers can easily skew the mean but not the median. Hence median becomes a better estimate than mean in this case.

### Retrival

While KNN is commonly used in supervised learning for classification and regression tasks, it can also be applied in unsupervised settings such as clustering. We use the idea of Nearest Neighbors for article retrieval. Each article is represented using TF-IDF representation. 

For example, if the user is currently reading article (<a href="https://www.cnbc.com/2025/06/03/chinas-may-factory-activity-unexpectedly-shrinks-clocking-its-worst-drop-in-nearly-3-years-caixin-.html">source</a>): `China’s manufacturing activity in May shrank at its fastest pace since September 2022, a private survey showed Tuesday, as a sharper decline in new export orders highlighted the impact of prohibitive U.S. tariffs.`. The following recommendations are generated:

```
You might also like:

Recommendation 1: UK House Prices Rise at Fastest Pace Since July (Update3) UK house prices unexpectedly rose in November at the fastest pace since July, reinforcing expectations real estate values will level out, avoiding a collapse from records, according to Nationwide Building Society.

Recommendation 2: UK growth at fastest pace in nearly 4 years Britain #39;s economy accelerated to the fastest annual pace in nearly four years in the second quarter as manufacturing emerged from a slump and consumers ratcheted up spending, the government said Friday.

Recommendation 3: China, ASEAN Agree to End Tariffs (AP) AP - China has reached agreement with the Association of Southeast Asian Nations, or ASEAN, on completely removing tariffs on merchandise goods by 2010 as part of a proposed free trade agreement, the Chinese Ministry of Commerce says.

Recommendation 4: Impact of euro played down The damage to exports caused by a stronger euro has been played down by a member of the European Central Bank #39;s governing council in remarks highlighting the bank #39;s limited concern about the currency #39;s rise.

Recommendation 5: Survey: Surge in layoffs, hiring Challenger survey finds most job cuts in 6 months; seasonal hiring by retailers lifts new jobs. NEW YORK (CNN/Money) - Employers increased both hiring and layoff plans in August, according to a survey released Tuesday by an outplacement firm.

```