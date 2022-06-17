# Supervised Learning

Supervised learning algorithms are trained using **labeled** examples. This means that we give it inputs where the desired output is known.

For example, we could give it a segment of text that could be categorized as *Spam* or as *Legitimate* for an email or *positive* or *negative* movie reviews.

This means that we already have the answer of our dataset.

**So How Does It Work?**

The network receives a set of input data along with the corresponding correct outputs and the algorithm learns by comparing its actual output with the correct outputs to find errors. Then it will modify the model accordingly.

**When Do We Use It?**

Supervised learning is commonly used in applications where historical data predicts future events.

## The process

1. Acquiring the data
2. Data cleaning
3. Splitting the data into Test Data and Training Data (+/- 70%)
4. Testing the model
   1. Adjust the model if required
5. Deploy the model

### Is it fair though?

Is it fair to use one set of data divided into training data and testing data and to evaluate the model based on that one split?

The data is usually split into 3 sets:

- The training data: Used to train the model parameters
- The validation data: Used to determine what model hyperparameters to adjust (more neurons, more layers, ...)
- The test data: Used to get some final performance metric

In other words, the test data is just to get an idea of how well the model performs.

## Evaluating performance

We just saw that after our machine learning process is complete, we will use performance metrics to evaluate how our model did. But we don't know how to evaluate yet.

 Let's start by analyzing the way to evaluate classification models, and later we will have a look at regression models.

### Classification

The key classification metrics are:

- Accuracy
- Recall
- Precision
- F1-Score

**What are those metrics and why do we use them?**

Typically, a classification task can only achieve one of the following two results: Either the prediction is **correct** or it's **incorrect**. This is also the case for multiple categories. For simplification let's assume a binary classification.

Let's imagine we want our model to determine whether a picture represents a dog or a cat. Since this is supervised learning, we will first **fit** (or train) a model on **training data**. This means that all the training data has already been labeled `dog` or `cat` - i.o.w. we already know if it is a cat or a dog. Then, we want to **test** the model on **testing data**. This means that we are going to provide the model with images it has never seen, and compare its predictions to the 'real' label of the image.

We will do this on a set of images and see how many the model got right and how many it got wrong. However, it is important to realize that ***in the real world, not all incorrect or correct matches hold equal value***.

In the real world a single metric won't tell the entire story. That's where our four classification metrics come into play. If we were to organize our predicted values (given by the model) compared to the real values (given by a human), it would result in what is called a **confusion matrix**.

#### Accuracy

The accuracy in classification problems is the number of correct predictions made by the model divided by the total number of predictions. This essentially gives us a percentage.

$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100\%
$$

The accuracy metric is useful when the target classes are well balanced. In our case, that would mean that there are roughly 50% of cat images and 50% of dog images in our set.

If we would have an **unbalanced** dataset, the accuracy would not be very interesting. Imagine we have 99% of dog images and that our model is not working properly, always predicting `dog`, no matter what.<br/>In this case, our *non-working* model would have a 99% accuracy despite being non-functional.

#### Recall and Precision

##### Recall

The recall is the ability of a model to find all the relevant cases within a dataset. More specifically, it is the number of *true positives* divided by the number of true positives plus the number of *false negatives*.
$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

##### Precision

The precision is the ability of a model to identify the relevant data points. More specifically, it is the number of *true positives* divided by the number of true positives plus the number of *false positives*.
$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

##### The trade-off

Often you will have a trade-off between **recall** and **precision**. While recall expresses the ability to find all relevant instances in a dataset, precision expresses the proportion of the data points our model says was relevant that were actually relevant.

#### F1-Score

The F1-Score is kind of a blend between recall and precision. In cases where we want to find an optimal blend of precision and recall, we can combine the two metrics using what is called the F1-Score.

It is calculated as the harmonic mean of precision and recal taking both metrics into account in the following expression:
$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$
We use the *[harmonic mean](https://corporatefinanceinstitute.com/resources/knowledge/other/harmonic-mean/)* over a simple average is because it punishes extreme values. For example, a classifier with a precision of `1.0` and a recall of `0.0` would have an F1-Score of  `0` but an average of `0.5`.

#### Confusion Matrix

A [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) looks something like this:<br/><img src="https://miro.medium.com/max/2102/1*fxiTNIgOyvAombPJx5KGeA.png" style="zoom:50%;" />

If we add all the formulas we've seen we will end up with a Matrix that looks like this:<br/>![](/home/nootaku/dev/data_science_with_python/ml_supervised/confusion_matrix.png)

#### In the real world

Often, as developers, we wonder if our accuracy is good enough or if our precision is good enough. But we need to keep in mind that we are developing for a purpose.

For example, if we want to create an algorithm that detects a disease, we need to understand that there is no disease that is equally spread among the population (meaning that 50% of the people have it, and 50% don't). This means that our initial data will not be balanced.<br/>According to this statement, accuracy is not the best metric to focus on.

However, if the disease is lethal, it might be interesting to calibrate our model in a way that there can be little to no false negatives. This might increase the amount of false positives, but it would be better to say *«Hey, sorry we thought you might have had it, but you don't.»* rather then *«Hey, sorry, we thought you didn't have it, but you do. And now it's too late.»*

### Regression tasks

Since we are not classifying things, but rather predicting a **continuous** values, it wouldn't make sense to use accuracy or recall metrics. We need to find metrics that are better suited to assess continuous values.

Let's start by imagining that we are predicting the price of a house given a set of features. This is the typical example of a **regression task**.<br/>If we were trying to predict the country the house is in given a set of features, that would be a classification task as there is only a finite amount of possibilities.

The most common evaluation metrics for regression are:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Square Error (RMSE)

#### Mean Absolute Error

The MAE basically consists of comparing the prediction to the actual expected value. We take the mean of the absolute value of errors.
$$
\frac{1}{n}\sum_{i = 1^{|y_i - \hat{y}_i|}} ^ n
$$
The problem of MAE is that it won't punish large errors.

#### Mean Squared Error

The MSE is, as the name implies, the mean of the squared errors. This has the advantage of making the large errors more prevalent than with MAE. And that's why it's more popular.
$$
\frac{1}{n}\sum_{i = 1^{(y_i - \hat{y}_i)^2}} ^ n
$$
The problem is that the MSE also squares the units of measure themselves. This means that if we want to predict the value of a property in €. We would end with MSE's in `€²`, which can be difficult to interpret 

#### Root Mean Square Error

The RMSE is the best of both worlds. By squaring the errors first, we ensure the value is positive. By doing the squareroot afterwards, we ensure that the value is not squared preventing us from having complicated units of measure.
$$
\sqrt{\frac{1}{n}\sum_{i = 1^{(y_i - \hat{y}_i)^2}} ^ n}
$$

#### The context

Just like everything else, the value of a RMSE, should be compared to the average value of the dataset we're working with.

For example: a 10€ RMSE is insignificant if we try to predict the prices of housing or rockets, but extremely significant if we are predicting the price of candy bars.

## Machine learning with Python

The main library for machine learning with python is `Scikit Learn`. It is very popular as many models are already present in the library. This document is purely theoretical so there won't be any code here, but in general, let's review how it works.

### Creating the model

Every algorithm is exposed in *scikit-learn* via an **Estimator**.

1. First import your model. The generic form is 

   ```python
   from sklearn.family import Model
   ```

   For example we would have `from sklearn.linear_model import LinearRegression` where `linear_model` is the family and `LinearRegression` is the model.

2. Set the parameters of the estimator. All the parameters of an estimator can be set when it is instantiated. Moreover, the default values are all suitable. For examplel: 

   ```python
   model = LinearRegression(
   	normalize=True
   )
   ```

   When printing the model, we can observe the parameters that were defaulted to the model such as the `copy_X` or the `fit_intercept` paramameters.

### Fit the model on some data

First thing to remember is to split our data into a training set and a test set.

1. Splitting our data: *(this is one way to do it)*

   ```python
   import numpy as np
   from sklearn.cross_validation import train_test_split
   
   # x is the data, y is the index for each row
   x, y = np.arange(10).reshape((5, 2)), range(5)
   
   # Split the data
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
   ```

2. Train (or fit) our model on the training data. This is done through the `model.fit()` method:

   ```python
   model.fit(x_train, y_train)
   ```

At this point our model has been fit and trained on the training data and is ready to predict labels or values on the test set.

### Predict new values

Since we are ready to predict values, lets try it out.

1. Use the predict method on the test values using the `model.predict()` method.

   ```python
   predictions = model.predict(x_test)
   ```

2. Evaluate our model by comparing our predictions to our `y_test` labels.

### Summary

The steps described above are available for all models when working with supervised estimators. Unsupervised learning works without labels so the `y_train` and `y_test` won't be available.

On supervised estimators, we also have access to the following methods:

- `model.predict_proba()`:  This will return the probability that a new observation has each categorical label. In this case the label with the highest probability is returned by `model.predict()`.
- `model.score()`: For classification or regression, scores are being returned between `0` and `1`. The larger score indicating a better fit.

## Choosing an Algorithm

Searching for `scikit-learn algorithm cheat-sheet` allows us to find this: ![](https://1.bp.blogspot.com/-ME24ePzpzIM/UQLWTwurfXI/AAAAAAAAANw/W3EETIroA80/s1600/drop_shadows_background.png)

This is basically a decision tree.

## Bias Variance Trade-off

