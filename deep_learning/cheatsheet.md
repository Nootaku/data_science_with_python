# Deep Learning - data science

This document is a guideline as to how to deal with real-world data and prepare a dataset for a Neural Network Model. It is purely theoretical but will contains tits and bits of code.

## Exploratory data analysis

### Check the balancing of the target label

#### Classification

In classification problems, it is a good idea to determine the percentage of points that are part of each category. This can be done by creating a `countplot`.

```python
sns.countplot(x='target_label', data=dataset)
```

There are 2 possible scenarios:

1. The distribution of points among the classes is relatively equal
2. The distribution of points among the classes tends to be mainly one category. In this case we are likely to have a very good **accuracy** ($\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} \times 100\%$), but the metrics we will have to look out for are the **recall** ($
   \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
   $) and the **precision** ($\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$).

If your classification (target label) contains categorical data (= strings instead of numbers), you will need to create a numerical classification. If this is a binary category, this would be `0` or `1`.

#### Regression

In regression problems we can create a `histplot` of the target label to have a general feel of the distribution.

```python
sns.histplot(data=dataset['target_label'], kde=True)
```



### Correlation between continuous features

I find it easier to read a `heatmap` than a pandas DataFrame.<br/>*Also, I prefer the `viridis` or `coolwarm` colormap for heatmaps.*

```python
correlation = dataset.corr()

plt.figure(figsize=12, 7)
sns.heatmap(data=correlation, annot=True, cmap='viridis')
```

#### High correlation present ?

If one or multiple features have a high correlation, then this needs to be explored further to prevent «leaking» data from our features into our label. In other words we want to make sure that there is not a feature that is a perfect predictor of our label.

### Correlation between the *"main"* feature and the label

Often there is one feature that sticks out as the _main_ feature. For example the price or the size of what we want to analyze. It is always a good idea to see if there is a correlation with this feature and our label.

#### Classification

In a classification problem we can `boxplot` this correlation.

```python
sns.boxplot(x='target_label', y='main_feature', data=dataset)
```



### Categorical features

When categorical features are involved, the first thing we want to do is to identify the *unique possible categories* independently of their population.

```python
dataframe['categorical_feature'].unique().values
```

A second good practice is to see if there is any correlation between the target classification and the categorical feature. This can be done by creating a `countplot`.

```python
sns.countplot(x='categorical_feature', data=dataset, hue='target_label')
```



## Data Preprocessing

### Dealing with missing data

When working with a dataset that has missing data, we have 3 options: *KEEP, DROP,* or *FILL* the missing data.

#### 1. Identifying missing data

Identifying the missing data means to know what we are dealing with. So first we check where we are missing data:

```python
dataset.isnull().sum()
```

And then we assess what percentage of our data is missing for each feature:

```python
(dataset.isnull().sum() / len(dataset)) * 100
```

#### 2. Deciding what to do

Once we know the proportion of our missing data we can make an informed decision regarding our 3 options. Obviously, if the missing data represents less than 1%, it is no problem to **drop** the missing rows. But for higher amounts we might consider **filling** or simply **keeping** the data as is.

- If we decide to fill we need to be careful about how to fill it. Sometimes we can extrapolate from the column itself, but often we need to extrapolate from a highly correlated feature.

  ```python
  dataset.corr()['feature_with_missing_values']
  ```

  Then we might have to make calculations based on those 2 features:

  ```python
  def myFunction(a, b):
      # do something
  
  df['feature_with_missing_values'] = dataset.apply(
  	lambda x: myFunction(x['feature_1'], x['feature_2']), axis=1
  )
  ```

  