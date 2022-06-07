# K Nearest Neighbour

 The ***K Nearest Neighbour*** algorithm is used for classification. It determines the classification of data based on the classification of the items that have features with values that are close to itself.

This means that this algorithm trades ease of implementation against precision.

## Scaling the data

Since the KNN determines the classification of an observation by identifying the items that are nearest to it, the scale of the variables actually matter a lot. That is because any variable that is on a large scale will have much more impact on the distance to the observation. Because of this, when we use KNN as a classifier, we want to **standardize everything to the same scale**. Luckily for us, `sklearn` has some built-in features for that.

```python
# The code that follows assumes you have a dataframe `df` with no excess columns

from sklearn.preprocessing import StandardScaler

# Create a scaler and fit it to the data (the scaler acts like a model)
scaler = StandardScaler()
scaler.fit(df)

# Actual scaling of the data of the dataframe `df`
scaled_features = scaler.transform(df)
```



## The Model

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

pred = knn.predict(X_test)
```

