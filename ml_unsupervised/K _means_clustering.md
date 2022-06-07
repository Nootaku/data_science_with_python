# K Means Clustering

K Means Clustering Algorithm allows to cluster unlabeled data in an unsupervised machine learning algorithm.

> As always the mathematics behind the K Means Algorithm (KMA) can be found in the **Introduction to Statistical Learning** at *Chapter 10*.

## Description

K Means Clustering is an **unsupervised learning algorithm** that attempts to group similar clusters together. This means it takes unlabeled data. 

It is typically used for:

- Market segmentation
- Clustering customers based on specific features
- The identification of similar physical groups
- Clustering similar documents

## How it works

First we choose a number of clusters `K`. Then, we randomly assign each point to a cluster.<br/>We then repeat a simple process until the clusters stop changing:

- For each cluster, we compute the cluster centroid by taking the *mean vector points* in the cluster
- We assign each data point to the cluster for which the centroid is the closest

<img src="/home/nootaku/dev/data_science_with_python/ml_unsupervised/kmc_process.png" style="zoom: 67%;" />

## Choosing a K value

Choosing the best K value is not an absolute knowledge. It is however better to chose a value for which we will not gain a lot of information if we were to increase K even more. One way to get to this result is the *elbow method*.

First we compute the sum of squared error (SSE) for some values of K (for example: 2, 4, 6, 8). The SSE is defined as the *sum of the squared distance between each member of the cluster and its centroid*.

Plotting K against the SSE, we observe that the error decreases as K gets larger. This is normal as the number of clusters increases, thus reducing the distortion of error.

The **elbow method** corresponds to choosing the value of K at which the SSE decreases abruptly.