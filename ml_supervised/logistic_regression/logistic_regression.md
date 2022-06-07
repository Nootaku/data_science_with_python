# Logistic Regression

Logistic Regression is a method for **Classification**. But what is classification ?

> In Machine Learning and Statistics, *Classification* is a problem of identifying to which of a set of categories a new observation belongs to based off of training data.

One type of classification is a Binary Classification meaning that there are only two classes an observation can belong to. Examples of Binary Classification are:

- Email: Spam vs 'ham'
- Disease Diagnosis (yes or no)
- Loan Default (yes or no)

The best known example of Logistic Regression is the exercise determining if a passenger of the Titanic survived or not based off of their features.

## Why logistic regression ?

The objective of logistic regression is to categorize an item. By definition, an item can either be part of a category or not be part of a category. This is why, by convention, we only have two classes for binary classification: `0` and `1`.

When we predict a classification, we predict a percentage (value between 0 and 1) to be part of a specific category.

Obviously, we can not use a linear regression model on a binary group as it wouldn't fit the data. This is because the training data that we will be using will either be part of the category (`1`) or not be part of the category (`0`) and it might lead the linear regression model to predict values below `0`, which is unusable for our purposes.

That's why we use a logistic regression curve instead since it can only return numbers between 0 and 1. It uses a *Sigmoid* function (see [here](https://en.wikipedia.org/wiki/Sigmoid_function)) that is calculated with the following function:
$$
\Phi(z) = \frac{1}{1 + e^{-z}}
$$
Since the *Sigmoid* Function will always be between 0 and 1, it means that we can feed our Linear Regression Solution and place it into the Sigmoid Function. This will alleviate the problem we had with some results being under 0 or over 1.
$$
linear = b_0 + b_1x \\
logistic = \frac{1}{1 + e^{(b_0 + b_1x)}}
$$
Finally, since we can now have a probability of if it is part of a category or not, we define that if that probability is over the threshold of 0,5 (or 50%) we consider that it is part of the category. Else not.

## Confusion Matrix

| `n =  165`     | Predicted: NO                                     | Predicted: YES                                    |
| -------------- | ------------------------------------------------- | ------------------------------------------------- |
| **Actual: NO** | True Negative (`TN`) = 50                         | False Positive (`FP`) = 10<br/>*aka Type I error* |
| Actual: YES    | False Negative (`FN`) = 5<br/>*aka Type II error* | True Positive (`TP`) = 100                        |

- Accuracy: How often is it correct ?
- Misclassification: How often is it wrong ?

## Actual code

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Create datasets for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    x,                # Feature data
    y,                # Target data
    test_size=0.3,    # Percentage of data that will be allocated to the test dataset
    random_state=101  # Randomness of selection of data allocated to the test dataset
)

# Create and fit model
logmodel = LogisticRegression(max_iter=10000)
logmodel.fit(X_train, y_train)

# Run predictions on test dataset
predictions = logmodel.predict(X_test)

# Create an automatic classification report
classification_report(y_test, predictions)
```

