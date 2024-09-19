---
title: "Pytorch vs MXNet: Which is faster?"
description: "An analysis of the computational efficiency of Pytorch and MXNet"
author: garethmd
date: 2024-07-18
categories: [benchmarking]
tags: [ai, beginner, ranking, pytorch, mxnet]
image:
  path: /assets/img/torch-vs-mxnet/banner.jpg
  alt: Pytorch and MXNet battle it out for for the runtime GC
---


I've written about GluonTs before and it's a great library for time series forecasting, however did you know that most of it's models are built on top of MXNet and for some architectures there is a Pytorch alternative. So I thought I would share some benchmarks I did to see which is faster, Pytorch or MXNet.

## MXNet
MXNet, originally developed by the Distributed (Deep) Machine Learning Community (DMLC), emerged from a collaborative effort to create a deep learning framework that balances flexibility, efficiency, and scalability. The project was spearheaded by prominent researchers such as Tianqi Chen, Mu Li, and other members of the DMLC group. Their goal was to design a framework that could seamlessly handle both symbolic and imperative programming paradigms, catering to a wide range of users from academia to industry. First released in 2015, MXNet quickly gained traction due to its strong performance in both single-machine and distributed environments. The framework's growth was further bolstered when Amazon Web Services (AWS) adopted it as the deep learning engine of choice, contributing to its development and integrating it deeply into their cloud services. This collaboration with AWS significantly amplified MXNet's visibility and adoption, and was used as the framework for many of AWS' projects including GluonTS, Gluon-NLP and Gluon-CV.

Since the late 2010's both industry and academia have gravitated towards Pytorch and Tensorflow and more latterly JAX. During this time we've seen many of the early forerunners of autograd frameworks like Caffe and Theano fall by the wayside and in 2023 MXNet became the latest to be retired. However it's legacy lives on in the GluonTS library which is what we'll be benchmarking today. 

## Pytorch
Pytorch is an open source machine learning library based on the Torch library which was written in Lua. Developed by Facebook, it offered a more pythonic interface compared to Tensorflow and was one of the first libraries to offer dynamic computation graphs, which meant that it was possible to dynamically interact with the model at run time rather than having to define the entire graph and compile it. Consequently it quickly became the darling of the research community where the ability to quickly prototype and debug models are an obbvious advantage to experiment with new ideas. It did however lack the outright performance of Tensorflow, but to some extent the two systems have converged in terms of performance and usability. However chances are that if you're reading a research paper related to deep learning then the code will be written in Pytorch.


## Setup
Lately I've been working on a project that uses the Monash Datasets and I've been using GluonTS to run both the MXNet and Pytorch versions of the DeepAR model. For each of the 40 datasets I have run each model 5 times. The parameters for each model are the same and the only thing that changes is the random seed. I'll record the total time taken for each model to run and then compare the two.
The time series datasets are pretty small as are the models so we're using a CPU to train the models.

## Results
The chart below shows the average time taken in seconds for each model to train across each of the datasets. 

![Runtimes](/assets/img/torch-vs-mxnet/runtimes.png)
*The average time taken (seconds) for each model to train across each of the datasets*

I was expecting Pytorch to be faster as it's had more development over the years but I wasn't quite prepared for the magnitude of the difference, but having watched my poor laptop struggle for an entire weekend to run the MXNet version of the model I can confirm that the results are accurate.

| Framework       | Runtime (s) |
|-----------------|--------------|
| MXNet Runtime   | 395807       |
| PyTorch Runtime | 88756        |

*The total time taken for each model to train across all the datasets*

In other words Pytorch is 4.5 times faster than MXNet.

## Conclusion
So if you're currently using MXNet for your deep learning projects then you might want to consider switching to Pytorch sooner rather than later. The performance gains are significant to make it worth the effort and you'll be on a framework that is actively being developed and supported.
