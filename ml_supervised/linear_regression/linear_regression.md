# Linear Regression

Linear Regression is the problem of predicting a **continuous value** of an item with specific features off of a training dataset with identical features.

> Official documentation for [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)<br/>Official documentation for [`LinearRegression` model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)<br/>Official documentation for the [`metrics`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)

For linear regression we are using ***SciKit Learn*** and its inbuilt models. In particular we will use the `model_selection` package for preparing our data and and the `metrics` model to analyze our model.

## Preparing the data for our model

Given a *DataFrame* `df`, we want to split `df` into two DataFrames: our features and our target.

```python
# Features
X = df[['list', 'of', 'columns', 'with', 'features']]

# Target
y = df['target_column']
```

Once we have our features and target, we need part of our data to be training data and part to be testing data:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,  # Feature data
    y,  # Target data
    test_size=0.4,  # Percentage of data that will be allocated to the test dataset
    random_state=101  # Randomness of selection of data allocated to the test dataset
)
```



## Create and train our model

Creating and training the model is very straightforward:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```



## Evaluate our model

Before we do any predictions, we can already evaluate our model. This is done with the `intercept_` and `coef_` methods.

```python
coef_df = pd.DataFrame(model.coef_, index=X.columns, columns=['Coefficient'])
```



## Create Predictions

Again creating predictions is very straightforward:

```python
predictions = lm.predict(X_test)
```

We can now compare our predictions to the actual target labels to validate our model. Sometimes this is also called analyzing or exploring residuals.

```python
# The scatterplot between the predictions and the actual target should be in a straight line
plt.scatter(y_test, predictions)

# A histogram with a kde should show a distribution that is as normal as possible
sns.displot(y_test - prediction, kde=True)
```

