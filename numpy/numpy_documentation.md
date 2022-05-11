# NumPy
NumPy is a Linear Algebra Library for Python. It is one of the main building blocks of most (if not all) the libraries in the PyData Ecosystem. It is also incredibly fast thanks to its bindings to C libraries.

***Installation***

```bash
# With conda
conda install numpy

# With pip
pip install numpy
```

## NumPy Arrays

> - [Link to official documentation](https://numpy.org/doc/stable/user/absolute_beginners.html#what-is-an-array)
> - [Link to the Jupyter Notebook](./numpy_arrays.ipynb)

Arrays are the central data structure of the NumPy library. An array is a grid of values and it contains information about the raw data, how to locate an element, and how to interpret an element. It has a grid of elements that can be indexed in [various ways](https://numpy.org/doc/stable/user/quickstart.html#quickstart-indexing-slicing-and-iterating) (one-dimensional, multi-dimensional, dots or iteration). The elements are all of the same type, referred to as the array `dtype`.

Each array has a `shape`. The  `shape` is a tuple representing the size of the array along each dimension.

> A **`vector`** is an array with a single dimension (there’s no difference between row and column vectors).<br/>A **`matrix`** refers to an array with two dimensions.<br/>A **`tensor`** refers to an array with three or more dimensions.

### Array attributes

An array is usually a fixed-size container of items of the same type and size. The number of dimensions and items in an array is defined by its shape. The shape of an array is a tuple of non-negative integers that specify the sizes of each dimension.

In NumPy, dimensions are called **axes**. This means that if you have a 2D array that looks like this:

```python
[[0., 0., 0.],
 [1., 1., 1.]]
```

Your array has 2 axes. The first axis has a length of 2 and the second axis has a length of 3.

### Reshaping arrays

It is possible to change the shape of an array using the  `reshape()` method. This method only works if the new shape of the array can host all elements of the array exactly. For example if there is one row and ten columns, it is possible to reshape to `(2, 5)` or `(5, 2)` but not to `(3, 3)`.

### Indexing arrays

Selecting data in an array is very similar to selecting data in lists. Just select the index: `array[index]`.

When working with multi-dimensional arrays, we use one integer to indicate the index of each dimension separated by a comma: `array[x, y, z]`.<br/>This means that when we need to select a *slice* of the array, we can use: `array[x1:x2, y1:, :z2]`

### No default copying of data

When we select a slice of an array, NumPy does not copy the data of the array into a new variable by default. The new variable is a pointer to the data in the original array. In other words, modifying the data in slice will modify the data in the array directly.

This default behavior can be overwritten with the `copy()` method.

```python
arr = np.arange(10)
arr_slice = arr.copy()[:5]
```

### Conditional selection

Perhaps one of the most used selection methods is the **conditional selection**. To do so, we select the elements of an array that have been submitted to a logical operation and returned `True`.

```python
# Create an array
arr = np.random.randint(0, 100, 10)

# Apply a logical operation to the array (creating a boolean array)
bool_arr = arr > 75

# Select the original array based on the boolean array
arr[bool_arr]

# This can be shortened to
arr[arr > 75]
```

### Operations on arrays

Is is possible to execute arithmetic operations on arrays. Simply use `+ 5`, `- 5`, `* 5`, `/ 5` to add, substract, multiply or divide each element of the array by 5.

***Warning:***<br/>Some operation (like `/ 0`) return an Error in Python but only a Warning in NumPy. For example:

```python
arr = np.arange(3)  # ==> [0, 1, 2]
foo = arr / arr		# ==> the first element should be 0/0, that is impossible

# This will result in a warning and the foo array being equal to:
# [nan, 1., 1.]
```
