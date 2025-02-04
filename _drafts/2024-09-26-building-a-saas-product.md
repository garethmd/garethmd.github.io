---
title: "Building a SaaS AI Product: An Architect's Perspective"
description: ""
author: garethmd
date: 2024-09-26
categories: [opinions]
tags: []
image:
  path: /assets/img/linear-ltsf/banner.jpg
  alt: 
---

## Introduction
Way back in 2005 I was involved in a project to select and implement a CRM system for a reasonably large business that I was working at. At that time I had a software web development background and thought the company that I was at was pretty progressive because it had Wifi in the office and we used a Data Center provider that we could serve our web-based applications on over a vpn. XML was pretty much the standard for data interchange and most apps were ASP.NET and served on IIS. We'd taken a look at Microsoft CRM and figured that we could deploy it if we mailed the installation CD's to our data center provider and then worked with them to setup SQL Server, and then one day, I can't remember how, I was introduced to Salesforce and I instantly knew that everything we were doing was going to be obsolete in a few years. No installation, no servers, no VPN, a SOAP API that you could call from anywhere, and a web-based interface that was configurable by the end user. It was a revelation.

Fast forward to 2018 I found myself in a greenfield project to build a SaaS AI product. The brief was as follows: create a multi-tenanted platform that can be sold to massive enterprise customers who will need to be able to configure the platform to their needs, setup users, create security settings, be easy to provision for each customer, easy to deploy and maintain and be cheap to run. What should it do? Well it should be able to read any data given to it by a user and allow support a machine learning model that could predict something useful using that data. It needed to be able to provide predictions in a real-time way and also in a batch mode... it needed to do all that and be damn good.

## Things we got right
## 1. The Cloud Platform
We knew that we needed a IaaS or a PaaS platform and one of the better decisions that I made was that we wouldn't prioritise portability, by which I mean, putting the code in a system that would allow it to be ported to another cloud provider. I figured that even if you did do that the chances of us actually migrating to another provider were so remote that it wasn't worth the effort. So the decision became what's the right platform for us. At the time GCP was around, but pretty limited and so the "safe" choice was to go with AWS. Fundamentally, I pictured myself having to have a meeting with a potential customer and going through their security and compliance requirements and being able to say "we're on AWS" and that would be enough, and sure enough that was actually a thing. 

## 2. Serverless
The next key decision that we made, was more of a philosphy and something that years of working with Salesforce had rubbed off on me. We would build the platform using, where possible, serverless technologies, and where not possible we would use the closest thing to serverless that we could find. Now at the time serverless was still pretty new and it came with two ket benefits: it was a pay as you go model meaning that we could keep our costs down and more accurately predict our costs based on usage which was a function of sales. Now somewhat disappointingly the term serverless has been extended to mean "no bare metal servers" and so you can have a serverless OpenSearch service that has a base cost and needs to be running all the time. The second benefit is security, because you're not running a server, there's no OS to patch, no ports to open, etc, etc, and I liked that because I wanted my engineers to be working on the product and not on the infrastructure, but at the same time I knew that if we built something that was insecure then we'd be finished.

## 3. The Capabilities
In retrospect I think we got this right more by luck than judgement, but when resources are tight it's imperitive to strike the right balance of unique capabilities and operational costs. Complexity costs, not just in development time, but in platform costs. Uploading a file to S3 once a week or even once a day and processing it a lot cheaper than implementing a real-time data pipeline that can handle even a modest number of requests per minute. Don't get dragged down a rabbit hole of building something you either don't need or can't afford to run. Keep it as simple as possible and 
try and architect for change so that you can add capabilities as you need them.

## 4. The Environment
One thing I think that's important to invest in early on is to create an automated development pipeline that will isolate development, testing and production environments and allow you to deploy changes on a regular basis. You don't need to go the whole hog running 2 week sprints and all that, but you should aim to put as much of your infracture into code as possible. We use serverless.com and seed.run to deploy our infrastructure and apart from a handful of manual steps we can deploy our entire platform with a couple of clicks. It saves time, it's more robust, it's more secure and it gives you all the tracking and auditing that you need to be able to see what's going on. Ulimately it allows you to lock down your production environment, because when people start doing manual changes in production that's when things start to go wrong. There's a lot of tools out there from Github Actions, AWS CodePipeline, Jenkins, SST, CDK, Terraform. Use them.


## 5. Don't get hung up on scalability
If I had to prioritise, certain things I'd say simplicity, maintainability,  security, reliability and then scalability. Obviously the thing that you're building needs to work for the demands you're going to expect, but don't start building in design for problems you don't have yet. If your lambda function is going to fall over processing a 1GB csv file that you're only going to run into once a year, then deal with it then. But trust me, if you start designing for scale from 
the outset, you'll either never get anything done or the thing that you'll scale for will be the wrong thing. We are terrible at predicting future requirements so don't try and just try to focus on building something that can be changed easily.

## 6. Thou shall enforce tennancy isolation
If you're building a multi-tenanted system then you need to make sure that you're isolating your tenants from each other. This isn't just a security thing, it's a will I still have a business if I don't do this right sort of a thing. There are some choices here. You can choose to create copies of your stack for each tenant, so that every customer has an individual copy of all the infrastructure. Now this is probably the easiest way to get started, but I've always been put off by the idea of having to manage lots of copies of the same thing. Imagine that you're pushing out updates and you're going to have to do it for each customer. That's a problem that's only going to get worse over time. The other extreme is to have shared rsources and use some logical partioning system to keep the data separate. In aws this isn't quite as bad as it sounds as there a reasonble amount of control with IAM that allow you to control access to DyanmoDB tables and S3 folders. This is the approach that we've taken and it's worked well for us. There's some initial investment in getting it right, but once it's setup then there's very little to do to maintain it, and everything else is just easier. Now there is a third way, where you can use Docker containers and arrange them in a way that are isolated from each other, but I'm not going to pretend that I know how to do that on the scale of a whole product. 


## 7. You shall honour thy clients data
I'm serious. If you ever find yourself logged into a production system updating customer data directly in whatever database your using, you need to stop and walk away. When you're architecting your system do so on the assumption that 
you won't be able to directly access the data. Scripts that can run as part of a planned process and can be tested in a development environment are fine, but don't be that guy who logs into the production system and runs a SQL script to and ends up dropping a bunch of tables - and yes I really have done that.



## Things we got wrong
## 1. Microservices
It's really easy to get carried away with microservices. They sound so great, we'll just create a tiny little services that's gonna live in it's own little world and it's going to consume and emit events like it's performing a ballet. The reality is that microservices add complexity and if you're not careful you'll find that it's really easy to end up with a bunch of shared dependencies that your microservices depend on, that innocent little lambda layer that going to provide some common shared code, as soon as you start doing that you've just tightly coupled all your services together. And when you're dploying a tiny change next you'll be wondering why 10 services need to be deployed. And whatever you please do not start writing logic in tons of different languages. I guarantee you that you'll end up having create duplicate code in different languages and you'll end up with a mess that will be impossible to maintain. By all means use microservices, but don't start with them, only add them when you need them.

## 2. Design Patterns, SOLID, TDD
I've noticed a trend in the year or so where there has been quite a backlash against Design Patterns and SOLID. I used to think any software engineer should be able to describe at least half of the GoF design patterns and have even gone as far as asking my engineers to present a pattern of their choice on a weekly basis. The argument against them is that it does something funny to your brain when you're writing and designing code. What used to come naturally all of a sudden becomes difficult because you're constantnly looking for the abstraction and the pattern that's going to fit your problem and I've come to the conclusion that it's true and instead I think a better approach is to write the simplest code that you can that solves the problem, avoid the temptation to abstract classes and just watch your code evolve. It's something I've been doing with my nnts library and for me at least it's worked better because I can move code about faster, I can change things more easily and I can see what's going on. We generally have a naturally tendency to abstract things far too early and it's a trap that constrains the design of your code, which will end up making it less flexible. 

## 3. Think really hard about SPA. 
Let's get one thing straight. I'm not a front-end engineer. Never have been and never will be. I don't have the patience for it and I don't have the eye for it. However, I have worked with a number of front-end engineers in my time, most of them have been react, and every single one has had a different view on how to structure a React project. Which is ironic because I thought the whole point of the damn thing was to enforce a structure and consistency. I see huge bloated react codebases for really very simple UX apps, tons of components. I just don't get it and in many ways I think it's a more complicated setup than the old days of using ASP or PHP. and I think it not just me that thinks like this with Carson Gross creating HTMX to try and get back to the days of simple web development and even Jeremy Howard (of FastAI) creating FastHTML. If you need to build a SPA, then think hard and allocate the right resources to it because it's going to be harder that you think.


## 2. Serverless
serverless OpenSearch service that has a base cost and needs to be running all the time. The second benefit is security, because you're not running a server, there's no OS to patch, no ports to open, etc, etc, and I liked that because I wanted my engineers to be working on the product and not on the infrastructure, but at the same time I knew that if we built something that was insecure then we'd be finished.


## 3. Monitoring
Sentry, seed

