# Seaborn

> [Official website](https://seaborn.pydata.org/)<br/>[Jupyter Notebook](./seaborn.ipynb)

Seaborn is a statistical plotting library built on top of **Matplotlib**. It provides a high level interface for drawing pretty looking statistical graphs.

The advantage of **Seaborn** is that it integrates very well with **Pandas** *DataFrame* objects. Most Seaborn methods are one-liners which makes it really convenient.

## Installation

```bash
# with Conda
conda install seaborn

# with pip
pip intall seaborn
```

## Plot types

### Distribution plots

[Distribution plots](https://seaborn.pydata.org/generated/seaborn.displot.html) are plots that evaluate the distribution of data points along one data series. Usually this corresponds to the columns of a data table.

There are different types of distribution plots:

- [Histogram](https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn.histplot): count of values within _bins_ and represents the result as a bar-chart
- [Rugplot](https://seaborn.pydata.org/generated/seaborn.rugplot.html#seaborn.rugplot): simple graphical representation along 1 axes of where the values of a series lie
- [KDE](https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot): sum of the Normal Distribution (= Gaussian distribution) of the rugplot.

```python
# default distribution plot:
sns.displot(dataset['column'])

# histogram:
sns.displot(dataset['column'], kind='hist')
sns.histplot(dataset['column'])

# rugplot:
sns.displot(dataset['column'], rug=True)
sns.rugplot(dataset['column'])

# kde:
sns.displot(dataset['column'], kind='kde')
sns.kdeplot(dataset['column'])
```

### Joint plots

Joint plots allow to graphically compare 2 distribution  plots. In other words it allows to visualize the correlation (or absence thereof) between two data series.

The documentation can be found [here](https://seaborn.pydata.org/generated/seaborn.jointplot.html).

A joint plot (`sns.jointplot()`) takes 3 arguments:

- `x`: Column name of the dataset to be compared as string
- `y`: Column name of the dataset to be compared as string
- `data`: The full dataset (not a series)

```python
sns.jointplot(x='column_1', y='column_2', data=dataset)
```

By default the joint plot is a ***scatter plot***, but just as for the `displot`, multiple kinds can be given such as: `hex` or `reg`. The latter is specifically interesting as it creates a regression line over the scatter plot.

### Pair plots

[A pair plot](https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot) is essentially a joint plot for every possible (numerical) column in a given dataset.

A pair plot can be further broken down in categories with the `hue` argument. This only takes categorical data, such as - for example - the sex of a person.

```python
sns.pairplot(dataset)

# With categorical breakdown:
sns.pairplot(dataset, hue='sex')
```

### Categorical plots

Categorical plots are focused on the distribution of data within **categorical columns** (such as the *sex* column we used here-above) in reference to either a numerical column or  another categorical column. The generic version of the plots that follow are `sns.catplot()`.

The three main types of categorical plots are:

- [Barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html#seaborn.barplot): aggregate data based on a function (`mean` by default)
- [Countplot](https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot): counting the occurrences of a category in a dataset
- [Boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot): show the distribution of categorical data
- [Violinplot](https://seaborn.pydata.org/generated/seaborn.violinplot.html#seaborn.violinplot): show the distribution of categorical data over a KDE plot. This is similar to the boxplot, but more detailed in exchange for a more complicated readability.
- [Swarmplot](https://seaborn.pydata.org/generated/seaborn.swarmplot.html#seaborn.swarmplot): a fusion between a strip plot and a violin plot

```python
# barplot
sns.barplot(x='sex', y='column_1', data=dataset)

# countplot
sns.countplot(x='sex', data=dataset)

# boxplot
sns.boxplot(x='sex', y='column_1', data=dataset)

# violinplot
sns.violinplot(x='sex', y='column_1', data=dataset)

# swarmplot
sns.swarmplot(x='sex', y='column_1', data=dataset)
```

### Matrix plots

Matrix plots allow to compare the relationship between different data series or dataset. It contains graphs such as [heatmaps](https://seaborn.pydata.org/generated/seaborn.heatmap.htm).

A heatmap represents relational data according to a gradient of color.

In order to create a heatmap, the data should already be in a ***matrix form***. This means that the index of a dataset should have a name. In other words, the integer-based system is not ok.

This can be achieved either by creating a correlation table (`pandas.corr()`) or a pivot table.

```python
# Loading a dataset
flights = sns.load_dataset('flights')

# Creating a matrix form with a Pandas pivot table
fpt = flights.pivot_table(index='month', columns='year', values='passengers')

# Heatmap
sns.heatmap(fpt)
```

## Styling

It is possible to set the general style of seaborn with the [`sns.set_style`](https://seaborn.pydata.org/generated/seaborn.set_style.html) method. The style can either be `white`, `ticks`, `whitegrid` or `darkgrid`.

### Size and Aspect ratio

Under the hood it works with the matplotlib `figsize`.

This means we can do the following:

```python
plt.figure(figsize=(12, 3))
sns.countplot(x='sex', data=dataset)
```

Another way to handle the size is the [`set_context`](https://seaborn.pydata.org/generated/seaborn.set_context.html) method. This allows to control the scaling of plots for a specific purpose. The context can be one of `notebook` (default), `paper`, `talk` or `poster`.

```python
# default use
sns.set_context('paper')
sns.countplot(x='sex', data=dataset)

# changing the font-size
sns.set_context('paper', font_scale=1.25)
sns.countplot(x='sex', data=dataset)
```

### Pallets and colors

We can set the colors of our `hue` or for our plot by using the `palette` argument and passing it a valid string. Those strings can be found [here](https://matplotlib.org/stable/gallery/color/colormap_reference.html).

```python
sns.countplot(x='sex', data=dataset, palette='Dark2')
```

