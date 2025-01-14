# MLE vs MAP

## Statistics

When predicting house prices, several factors influence the final value, such as the neighborhood, the number of rooms, square footage, age of the house, and so on. Some of these factors, like the neighborhood, are complex and difficult to model accurately. However, one way to simplify this problem is by assuming a linear relationship between the house price and easily measurable features, and assume a gaussian noise.

<img src="../img/linear-gaussian.jpg" alt="linear-regression">

This idea of breaking a complex process of house price prediction into a simple linear process and gaussian noise is central to statistics. In other words,

```
Complex Process = Simple Process + Noise
```

The challenge lies in carefully modeling the right features (the simple process) and accurately modeling the noise, which might require domain knowledge.

In simple words, statistics aims to create models or estimate parameters that best explain the data we have observed. Maximum Likelihood Estimation (MLE) and Maximum A Posteriori Estimation (MAP) are both statistical methods. 

### MLE vs MAP

In MLE, we simply look at the data collected and find the model which generated the data with maximum likelihood. This works well when we have no prior knowledge. In MAP we take the same likelihood we used in MLE but now multiply by our prior knowledge. This new estimate is a mixture of what we believe (our prior) and what we measured (our likelihood).

Imagine studying a rare species of plant, where we want to estimate the average height of fully grown plants in the species. We have collected data for the heights of 20 such plants. We dont know the distribution of plants heights, but we can model it using gaussian distribution. 

Now we don't have any prior information on the plants’ average height. Thus, by MLE, we try to find a gaussian distribution which will maximize the likelihood of data we have observed. Thus, the estimator mean of gaussian is equal to the sample mean and the estimator variance of gaussian is equal to the unadjusted sample variance. For instance, we may strongly suspect that our coin is biased and so we can influence our estimate with that knowledge through a prior distribution. 