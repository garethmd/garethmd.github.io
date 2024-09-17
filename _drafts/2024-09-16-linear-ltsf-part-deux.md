---
title: "LTSF-Linear: Embarrassingly simple time series forecasting models"
description: "A review of the 2022 Paper Are Transformers Effective for Time Series Forecasting that introduced DLinear and NLinear models"
author: garethmd
date: 2024-09-16
categories: [research]
tags: [dms, linear, time-series, forecasting]
image:
  path: /assets/img/linear-ltsf/banner.jpg
  alt: Linear models go head to head with transformers like David vs Goliath
---


[Are Transformers Effective for Time Series Forecasting](https://arxiv.org/pdf/2205.13504) by Zeng et al. They suggest  All the code for this post can be found on my [nnts](https://github.com/garethmd/nnts/tree/benchmarking) github repo.

# Multivariate vs Univariate
Primarily the paper is interested in the performance in a multivariate setting as the majority of the transformer architectures that it compares itself to are intended for multivariate time series forecasting. So I want to 
take a moment to describe exactly what we mean by Multivariate and Univariate time series forecasting.

Most time-series forecasting datasets will comprise of a single file containing multiple time series. For example the Tourism Monthly dataset has 366 individual time series, one for each region in Australia, New Zealand and Hong Kong. Each time series is a sequence of monthly observations and the number of observations in each series may be different as the historical data may start at different times in each region. When we forecast we aim to predict the future values for each of the time series across a forecast horizon, but because the time series in the dataset do not share a common time span we do not attempt to model the relationships between each time series, so we forecast each time series independently of the others. This is what we refer to as Univariate. Our input is a single time series and our output is a single time series.

By contrast Multivariate forecasting is where we have multiple timeseries that do share a common time span and we aim to 
model the association between the time series. This means that the input to the model is essentially a table of values containing a sequence of historical values for each time series in the dataset. Obviously this requires the dataset to be structured in such a way that all the time series are aligned and typical examples of this are ETTh, Electricity and Traffic datasets.

In summary not all datasets are suitable for multivariate forecasting, but all datasets are suitable for univariate forecasting.

# Local vs Global
The authors discuss Univariate forecasting and present the results in the appendix, but here is where I start to have a problem with the setup of the experiments they performed and in order to understand why we need to talk about the concept of local and global forecasting.

Local forecasting is where we create a model for just one time series. So in our Tourism Monthly example we would create 366 individual models (one for each region) and then forecast the future values for each region independently. This is a perfectly reasonably approach but has some limitations, for example if we obtain a new time series for a new region we would have to train a new model and secondly as the number of time series increases the number of models we have to train increases linearly. Depending on the use case we may have tens or hundreds of thousands of model artifacts to manage. 

Global forecasting attempted to address this, by creating a single model that can forecast all the time series in the dataset. This is more scalable and most if not all neural network univariate models are global models and report their results as such.

DLinear follows a trend in recent multivariate forecasting research which takes an approach that is neither local nor global. Firstly, the model architecture can be configured to operate in a "univariate mode", referred to as channel independence. Essentially this means that a set of linear layers are created for each time series in the dataset. The weights of these layers are dedicated to one time series and are not shared. Because the model only has 
one layer it means that none of the learnable parameters are shared between the time series. To all intents and purposes this is a local model. However, everything else about the model is global. We sample all the time series in each training example, we calculate the loss across all the time series and we update the weights of the model based on the loss. In this univariate setting the model has a bit of an identity crisis. 

What really concerns me about this approach, that nobody seems to have thought about is the impact this has on performance. In this channel independent configuration we no longer have one big matrix calculation for all the time-series as would be the case in the multivariate setting, but instead we have lots of little matrix calculations (two for each time-series in the dataset). Now that may not be an issue if there are few hundred time-series in the dataset, but when have tens or hundreds of thousands of timeseries then performance just grinds down to a crawl. At scale this approach just does not work. We will see just how much of an issue this is later, but in my opinion this has not been addressed in the paper, in part because of the way that the univariate experiments are run which we will discuss next. 

The univariate results are conducted using the ETT dataset, which has 7 time seies, and following the same methodology as many of the transformer papers which uses just one time series from the dataset to train and evaluate the model. That is to say one time series is selected, in this case the "OT" feature, all of the other series are discarded and the model is trained and evaluated on this one time series. This makes it equivalent to a local model. In other words this has been run as a univariate local model and not a univariate global model and as such the results are not directly 
comparable to global models. Oh and by the way the authors are not the only offenders here, the same methodology has been used in many of the multivariate transformer papers.

# Global, Local, Independent and Multivariate experiments
So you maybe asking yourself why does any of this matter? Who cares if this is a local or a global model. Well it matters because when we use these models as benchmarks we need to ensure that we are comparing like for like. So in order to do this I have run the DLinear and NLinear models in the following configurations:

- Local - One model for each time series  
- Independent - One model but with no shared weights  
- Multivariate - One model for all time series forecasting all time series simultaneously  
- Global - One model for all time series forecasting each time series independently (To achieve this the models were set to have one channel and then trained using a univariate data sampling strategy)

The datasets that we'll be using are from the Monash datasets which I've used in previous posts are and are a collection of 40 univariate and multivariate time series datasets. The forecast horizons are diverse and come from different domains. For brevity I'll just present the results from 6 datasets.


## Computational Efficiency
The first thing we'll look at is the computational efficiency of the models. *Table 1* below shows the total training time in seconds for 100 epochs for each configuration on the 6 datasets.


| Model               | Local     | Independent | Multivariate | Global  |
|---------------------|-----------|-------------|--------------|---------|
| Car Parts           | 446.966   | 124.398     | **6.175**        | 11.325  |
| COVID               | 203.772   | 32.353      | 8.806        | **8.735**   |
| Electricity Hourly  | 2521.712  | 1685.107    | 240.695      | **11.053**  |
| Electricity Weekly  | 137.398   | 23.509      | **5.501**        | 9.339   |
| Traffic Hourly      | 7098.796  | 13801.659   | 462.455      | **12.987**  |
| Traffic Weekly      | 214.614   | 29.594      | **5.048**        | 9.484   |

*Table 1 DLinear model total train time in seconds for 100 epochs. Shortest training times are shown in **bold***

As we can see the global and multivariate configurations train far more quickly than our local configuration or the channel independent setting. In the extreme case the Traffic Hourly dataset takes 3 hours 10 minutes to train with channel independence, but just 12 seconds in the global configuration. 

The fact that there is such a marked difference between Independent and Multivariate proves that performance is dominated by the number of of separate matrix calculations that need to be performed and not by the total number of trainable parameters.

## Forecast Accuracy
So now let's turn our attention to how well each configuration performs in terms of forecasting accuracy. Maybe the performance of the local and indepedent models are so good that they justify the extra computational cost. *Table 2* shows the result of the Mean Absolute Scaled Error (MASE) for each configuration on the 6 datasets.  The best error for each data set is shown in bold and the second best in underlined.


| Model             | Local  | Independent | Multivariate | Global |
|-------------------|--------|-------------|--------------|--------|
| Car parts         | 1.271  | 1.403       | <u>0.752<u>        | **0.747**  | 
| Covid deaths      | 9.354  | 9.200       | **5.601**        | <u>5.974<u>  |
| Electricity hourly| <u>1.881<u>  | 1.883       | **1.880**        | 2.012  |
| Electricity weekly| 1.049  | 1.096       | **0.780**        | <u>0.827<u>  |
| Traffic hourly    | 0.965  | 0.968       | <u>0.923<u>        | **0.918**  |
| Traffic weekly    | 1.454  | 1.487       | **1.096**        | <u>1.130<u>  |

*Table 2 MASE for each configuration tested. The best error is shown in **bold***

 Unsurprisingly the Local and Independent results do appear to be closely related, however there also seems to be an association between the Multivariate and the Global results. For example the two best performing configurations for Car parts are Multivariate and Global both with a MASE of ~0.75 and Local and Independent have similar errors of ~1.3 and ~1.4 respectively. NLinear has the same pattern of behaviour. Clearly the characteristics of certain datasets tend to benefit from sharing information between time-series. These results seem to suggest some evidence that for these Linear models information can be shared using univariate or multivariate methods to a similar effect.  Given the limitations of multivariate models it would be interesting to see if this observation holds for other multivariate models.



# Conclusion
So there we have it a look at the LTSF Linear models and how they perform in different configurations. Now these mulivariate models may outperform everything else once you start using forecast horizons in excees of 336 time steps, but for anything shorter than that it seems like univariate models still have the edge. For me the key question is given the additional awkwardness and limitations of multivariate models is that really the best approach? The evidence here suggests that with these models at least the answer is probably not. I would prefer to use a global model that I know will scale and can be used on any dataset not just those that are suitable for multivariate forecasting.
