# Principal Component Analysis

> More information about Principal Component Analysis can be found at chapter 12.2 of Introduction to Statistical Learning.

## Description

Principal component Analysis (or PCA) is an unsupervised statistical technique used to examine the interrelations among a set of variables in order to identify the underlying structure of those variables. It is also known as **factor analysis**.

We have seen regression. Regression determines a line of best fit to a dataset. PCA, however determines several *orthogonal lines* of best fit to a dataset. Orthogonal means at "a right angle". So the lines determined by the PCA are lines perpendicular to each other in an *n-dimensional space* where `n` is the number of variables of the dataset.

## What is it used for ?

PCA is usually used on big datasets with a lot of features. When dealing with those datasets it can be complicated to find which feature is important to our analysis and which feature is noise. This is called the **curse of dimensionality**. This problem is also exacerbated when the different features (or dimensions) don't have the same scales.

The role of PCA is to reduce the number of dimensions by choosing the features that are the most important to the linear relationship of the dataset. These features are called **components**. Components are a linear transformation (of a combination of features) that chooses a variable system for the dataset such that the greatest variance of the data comes to lie on the first axis. The second greatest variance will come to lie on the second axis and so on.

This allows us to compress or reduce the number of variables used in an analysis.

## When should we use PCA?

We should try to reduce or change dimensions when we are aiming for better perspective and less complexity. In other words when we need a more realistic perspective and we have many features on a given dataset of which we know that we don't need this many features.

We also use PCA when we aim at a better visualization of our data. When we cannot get a good visualization due to the high number of dimensions, we are trying to "flatten" our dataset.

It also allows us to reduce the size of our dataset when we have too much data and we are planning to use process-intensive algorithms. PCA will help us get rid of redundancy.

