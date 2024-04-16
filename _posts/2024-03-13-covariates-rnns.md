---
title: "Time series forecasting with RNN's and covariates"
date: 2024-03-13
categories: [time-series, rnn, auto-regressive]
tags: [seq2seq, rnn, lstm, gru, auto-regressive]
image:
  path: /assets/img/teacher-forcing/banner.jpg
  alt: image alternative text
---
## 1. Introduction  
Leading indicators are commonplace in business. Knowing how many opportunities there are in a sales pipeline today signals how much business is likely to be closed in the future. During the pandemic, infection rates from tests was used to model future hospitalisation rates. I find it surprising that there is very little support for these indicators or covariates in time-series models. Covariates are typically expetected to be deterministic ie you have to know them in advance for the timesteps that they are applied to. Now this is fine if it's something like a date property that we can always calculate, but what if we don't know with any certainty what the values of the covariates will be in the future? This was briefly discussed in DeepAR [1] where the authors posited that you could either simply copy the most recent known past covariate value into future timesteps or you could predict the covariates along with the target variable from each timestep as a vector. It is the latter approach that I will explore in this post.


## 2. Teacher Forcing

In code this is a very natural thing to do as we can set the target to be the feature vector of a sequence shifted right by one time step.

```python
data = torch.rand(10, 5, 3) # 10 sequences of length 5 with 3 features
X = data[:, :-1, :] # remove the last time step
y = data[:, 1:, :] # shift the target right by one time step
```

## 3. Free Running


## 4. Scheduled Sampling / Curriculum Learning

![Curriculum Learning](/assets/img/teacher-forcing/CurriculumLearning.jpg)  



## 5. Professor Forcing

![Professor Forcing](/assets/img/teacher-forcing/ProfessorForcing.jpg)  


## 6. Attention Forcing


## 7. Final Thoughts



## References
1. [DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://arxiv.org/abs/1704.04110)


