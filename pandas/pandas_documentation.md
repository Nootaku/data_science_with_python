# Pandas
> Official documentation: [here](https://pandas.pydata.org/docs/)<br/>Jupyter Notebook: [here](./pandas.ipynb)

Pandas is a library built on top of NumPy. It allows for fast data analysis, data cleaning and data preparation. In other words it is an excellent tool for performance and productivity.

Pandas also has built-in features for data visualization and can work with a wide variety of sources including `.xls`, `.xlsx`, `.csv` or `.json`.

## Installation

```bash
# with Conda
conda install pandas

# with pip
pip install pandas
```

## Series

A series is an object similar to a NumPy array, but has an feature that allows the elements to be accessed by label.

It is a one-dimensional ndarray with axis labels (including time series).

Labels need not be unique but must be a hashable type. The object supports both integer- and label-based indexing and provides a host of methods for performing operations involving the index. Statistical methods from ndarray have been overridden to automatically exclude missing data (currently represented as NaN).

It can take a list, NumPy array or dictionary as input.

```python
ser = pd.Series(
	data=my_data,
    index=my_labels
)
```

## DataFrames

A dataframe is an object resembling a table. It is a two-dimensional, size-mutable, potentially heterogeneous tabular data.

Data structure also contains labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. The primary pandas data structure.

```python
data = np.arange(25).reshape(5, 5)
df = pd.DataFrame(data)
```

### Working with DataFames

A lot of inbuilt methods exist for DataFrames (= df) such as `sum()`, `mean()` or `std()`.

But it is also possible to create your own functions and to apply them to your df with the `apply` method.

```python
def double(x):
	return x * 2

df['column_n'].apply(double)
```

This will actually run double on each value within the `column_n` of the df. This can be further simplified by using a lambda function:

```python
df['column_n'].apply(lambda x: x * 2)
```

#### Useful inbuilt methods

A few of the most useful inbuilt methods are:

- `unique`: creates a list of unique values in the range
- `nunique`: returns the amount of unique value in the range (equivalent to `len(df['col'].unique())`)
- `value_counts`: returns a list of unique values and their occurrence count
- `sort_values`: takes at least one argument `by` corresponding to a column name or list of column names and will reorder the rows of the df based on the values of this (these) column(s).
- `pivot_table`: advanced function that returns a multi-indexed df based on the given arguments ([see documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot_table.html?highlight=pivot_table#pandas.DataFrame.pivot_table))

## Input Data

Pandas can read and write data from and to a large variety of sources. Here I'm interested in `CSV`, `Excel`, `HTML` and `SQL`.

Of course some dependencies are required for each source:

```bash
# using Conda
conda install sqlalchemy
conda install lxml
conda install html5lib
conda install BeautifulSoup4

# using pip
pip install sqlalchemy
pip install lxml
pip install html5lib
pip install BeautifulSoup4
```

`sqlalchemy` is required to handle SQL, `lxml` for XML-based sources such as Excel and `html5lib` for parsing HTML. `BeautifulSoup4` sits on top of  `lxml` and `html5lib` and allows easy pythonic methods for iterating, searching and modifying parse trees.
