---
title: "Supercharge your Neural Networks with Super-convergence"
description: "A look at 1Cycle scheduling, one of my favourite techniques at improving model performance and practical guidance on how to use it"
author: garethmd
date: 2024-06-12
categories: [neural-networks, training]
tags: [neural-networks, training, super-convergence, scheduling, learning-rate]
image:
  path: /assets/img/superconvergence/banner.jpg
  alt: superman does superconvergence
---
## 1. Introduction 
Want to know how to get a performance boost from your neural network for virtually no extra effort? Well I'm going to show you how to do it 
with a learning rate scheduling system called 1Cycle which leads to a phenomenon called super-convergence. I find that it improves test accuracy and generalisation
so reliably that I use it in all my projects and have done so for years.

Neural networks can be sensitive beasts and getting them to train well can be a bit of a black art. One of the most important hyperparameters to get right is the learning rate  which regulates the magnitude of the updates that are applied to the weights of the network. Too high and the model will diverge and too low and the model will train very slowly and will likely settle in a local minima which will give sub-optimal performance. Adaptive learning rate optimisers such as Adam, Adagrad and Adadelta which modify the standard SGD optimisers to dynamically change the size of the updates based on the properties of the gradients help to alleviate this issue and now in my experience virtually all networks use Adam or one of its variants. Even with these improvements it is still common to use a scheduler to adjust the learning rate during training to help the model converge.

A conventional approach to learning rate scheduling is to reduce the learning rate over time by some factor at regular intervals or when the validation loss stops improving. A few years ago there was a bit of a fashion for creating learning rate schedulers each with its own profile of how the learning rate would be changed over time. You didn't have to do something as dull as just linearly reducing the learning rate, you could do exponential decay, polynomial decay or any function really with it's own characteristics and *figure 1* shows just a few.

![Pytorch Schedulers](/assets/img/superconvergence/Schedulers.png){: width="900" }
*Figure 1. The StepLR, Cosine Annealing with Warm Restarts and CycleLR schedulers learning rates during training*

Nowadays the trend seems to be using ReduceLRonPlateau which reduces the learning rate when the validation loss stops improving. 



## 2. Super-Convergence
There is, however, another way that I discovered some years ago watching one of Jeremy Howard's excellent series of videos on [Practical Deep Learning for Coders](https://www.youtube.com/@howardjeremyp/playlists){:target="_blank"}. A way that was proposed by Leslie N. Smith and Nicholay Topin in their 2018 paper [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/pdf/1708.07120){:target="_blank"}.

Leslie, a researcher with the US Navy, is in my view a great engineer and experimentalist and doesn't get the recognition that he deserves. He gave a great talk at [ML Conf Competition Winning Learning Rates](https://www.youtube.com/watch?v=bR7z2MA0p-o&ab_channel=MLconf){:target="_blank"} where he talks about the work in the paper which I highly recommend, but I'll give you a brief summary of his talk and his paper here. 

So the story goes that Leslie was interested in the effect of learning rate on the training performance and was experimenting with different techniques to find the best learning rate. At the time the standard way of doing this was by running a series of experiments with something like a Grid search and choosing the rate that gave the best results. He found that if he used a cyclical learning rate (CLR), shown in the far right plot on *figure 1*, then he would get the same performance as if he had just used the optimal learning rate. The CLR is a system which varies the learning rate between two bounds during training and he would aim to set the range so that the optimal learning rate was somewhere close to but below the upper bound.

Leslie says that something interesting came out of these CLR experiments. He found that if he varied the learning rate between the two bounds and then plotted the accuracy of the model against the learning rate then the accuracy would reach a peak which corresponded to the optimal learning rate. The great 
thing about this was that you didn't need train the model to convergence, but you could just run a small test to find the optimal learning rate. He called this the LR Range Test.

![Alexnet LR Range Test](/assets/img/superconvergence/normalRangeTest.png)
*Figure 2. LR Range Test for AlexNet on the ImageNet dataset from [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/pdf/1708.07120){:target="_blank"}*


So, job done right? Well not quite. Back then all the cool kids were doing computer vision and the trend was very much towards developing deeper model architectures which were more capable at extracting abstract features that you find in images and therefore produced better results. CNN model architectures like Resnet and Inception, with their 34+ layers and residual connections were the state of the art at the time. Now, when Leslie ran the LR Range Test on them he found that they produced no peak even when he used a ridiculously large range up to a learning rate of 3, which was about an order of magnitude larger than anything reasonable.


![Resnet Inception LR Range Test](/assets/img/superconvergence/range3Res56.png){: width="500" height="500" }
*Figure 3. LR Range Test for ResNet on the ImageNet dataset from [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/pdf/1708.07120){:target="_blank"}*


Wondering what was going on he decided to run an experiment with just one cycle starting with a low learning rate of 0.1, increasing it up to 1 and then decreasing it back to the low value of 0.1. He found to his surprise that the model trained about an order of magnitude faster than by the standard approach of stepping down the learning rate. Not only that but the test accuracy was actually better than with the traditional approach. This effect he called Super-convergence and the key thing to note here is that the peak learning rate is way higher than what you would typically use. He then found that you could increase accuracy further still by decreasing the learning rate to a value below the initial learning rate with a long tail. 

![Superconvergence](/assets/img/superconvergence/imagenetResnetSC.png){: width="500" height="500" }
*Figure 4. Super-convergence for ResNet on the ImageNet dataset from [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/pdf/1708.07120){:target="_blank"}*

During the course of his experiments he found that using a 1Cycle policy had the most beneficial effect on training datasets that had limited data, and that may 
in part explain why I have had such good results with it as in the field of time series and tabular data where I spend most of my time we tend to have relatively small datasets.

Leslie makes an interesting observation about changes to other hyperparameters that are required when using 1Cycle to keep the whole system "balanced". He states that using a larger learning rate is a form of regularisation in itself and as a result he found that he would get the best results when he reduced other 
forms of regularisation such as weight decay, dropout and batch size.

That covers the main points of the paper with the exception of a section regarding an approach for estimating the optimal learning rate which he uses
to demonstrate that large learning rates are effective at finding good minimums in flat loss landscapes. This section is a bit mathy math and it's really not essential to the main points of the paper so I'll leave to you to read if you're interested. Overall, it's a great paper and if you've never read a research paper before then I would highly recommend it as it's very accessible and well written.


## 3. How I use it
Now if like me you use Pytorch then you're in luck because it has a built in scheduler called [OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR) which implements the 1Cycle policy. If you're using a different framework like Tensorflow or Jax then you'll have to implement the scheduler yourself, but it'll be worth the effort.

![Pytorch 1Cycle Schedule](/assets/img/superconvergence/Pytorch1Cycle.png){: width="500" height="500" }
*Figure 5. The 1Cycle learning rate schedule profile as implemented in Pytorch*


It's really easy to use. Essentially you initialise the scheduler by passing the optimizer, the maximum learning rate and the number of steps in an epoch. Typically I will set the maximum learning rate to be 3 times the learning rate that I would normally use. 

```python
lr = 0.1
data_loader = torch.utils.data.DataLoader(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*3, steps_per_epoch=len(data_loader), epochs=10)
for epoch in range(10):
    for batch in data_loader:
        train_batch(...)
        optimizer.step()
        scheduler.step()
```

Unlike most other schedulers in Pytorch you call the scheduler every batch rather than in every epoch. There are a number of other parameters that you can set but I've never found the need to change them. Interestingly, in the appendix of the paper Leslie comments that he found that he was unable to obtain super-convergence with Adam, but I've never had any issues with it. That may be because I always use the AdamW variant, but I'm really not sure and haven't investigated it further.

Also I know that the paper goes into some detail about reducing other forms of regularisation, but I have to admit that I generally don't do that, and I prefer to
keep the other hyperparameters the same as an initial run and then tweak them if I don't get the performance that I'm looking for. Generally however I find that I get good results just by replacing the standard scheduler with 1Cycle and keeping everything else the same.

I find that I get two benefits: firstly absolute performance measured on the test set is improved, secondly training stability is improved and the variance of the test performance metrics is reduced. I've tried lots of different schedulers over the years and I always come back to 1Cycle. 


## 4. Conclusion
So there you have it a look at 1Cycle scheduling, one of my favourite techniques at improving model performance and some practical guidance on how to start using it. In the next post I'll show you a real-world example where using 1Cycle scheduling produces real state of the art performance. With that I will leave you with some words from the great man himself: 

"The moral of this story is: curiosity, enlightenment, continual learning.. it's important to all of you. You should all keep on learning, always."

