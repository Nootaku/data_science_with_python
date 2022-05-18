# Matplotlib

>Official Documentation: [here](https://matplotlib.org/stable/users/index.html)

 Matplotlib is the most popular plotting library for Python. It allows control over every aspect of a figure and was designed to have a similar feel to MatLab's graphical plotting.

One of the most interesting pages of the Matplotlib documentation is the [gallery](https://matplotlib.org/stable/gallery/index.html). It contains a lot of examples of what is possible with Matplotlib.

## Installation

```bash
# installing with conda
conda install matplotlib

# installing with pip
pip install matplotlib
```

## Plotting

### Two systems

Often we are going to use the Jupyter Notebook system to visualize our data. In order to allow Jupyter to visualize the data we need to add `%matplotlib inline` at the top of our workbook.

```python
import matplotlib.pyplot as plt
%matplotlib inline
```

If we want to visualize our plots when we are not using Jupyter Notebook. We will need to finish our plotting code with:

```python
plt.show()
```

### Two ways

There are two ways to create a Matplotlib plot:

- the functional method
- the object-oriented method

The latter is recommended, especially when working with more complex plots.

### Object oriented method

The object oriented method means that we are going to instantiate a figure object. And then we are going to call methods on that object.

```python
# Creation of a blank canvas
fig = plt.figure()

# Adding axes to the figure
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# Plot data on the axes
axes.plot(x, y)
```

## Subplots

Using the Object-Oriented method, we can create subplots by writing:

```python
fig, axes = plt.subplots()
```

## Save a figure to file

```python
fig = plt.figure()
fig.savefig('my_picture.png')
```

