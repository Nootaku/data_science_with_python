# Plotly and Cufflinks

> Official documentation [for Plotly](https://plotly.com/python/)<br/>and [for Cufflinks](https://github.com/santosjorge/cufflinks)
>
> A cheatsheet is available [here](./plotly_cheat_sheet.pdf).
>
> Our Jupyter Notebook is [here](./plottly.ipynb).

Plotly and cufflinks are libraries that allow to create interactive visualizations that are integrated with Pandas.

Plotly is the interactive visualization library, and Cufflinks is the library that connects plotly to pandas.

## Installation

```bash
# Install Plotly
# with Conda
conda install -c plotly
# with pip
pip install plotly

# Install Cufflinks
conda install -c conda-forge cufflinks-py
pip install cufflinks
```

### First steps

Let's get the elephant in the room out of the way. Plotly is not easy. More often than not, we'll need a reference code and and adapt the values to fit our needs. However, Cufflinks makes it easier.

Cufflinks allows to create Plotly graphs using the syntax of the built-in Pandas data-visualization methods. Here is how that works.

Start by copying these complicated imports:

```python
import pandas as pd
import numpy as np
import cufflinks as cf

# Since plotly has a paid online version, we need to specify that we want the offline open source api
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

# Then we need to initiate the notebook mode (since we work with pandas)
init_notebook_mode(connected=True)

# And indicate to Cufflinks that we are offline.
cf.go_offline()
```

You can then use the `df.iplot()` methods (see Plotting).

### Encountered problems

The documentation I was following might have been a bit dated (2018) leading to some errors popping up. Here is what happened and how I solved it:

#### Import Error

The name of the main library `plotly.plotly` has changed and is now `chart_studio`. The documentation for Chart Studio can be found [here](https://plotly.com/python/getting-started-with-chart-studio/).

The error message:

```
ImportError: 
The plotly.plotly module is deprecated,
please install the chart-studio package and use the
chart_studio.plotly module instead. 
```

The fix:

Import `chart_studio.plotly` instead of `plotly.plotly`.

```python
import chart_studio.plotly as py
import plotly.graph_objects as go

# If we want to work offline:
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
```

#### Jupyter lab vs Jupyter notebook
When installing the pip version of Cufflinks, Plotly does not work properly on Jupyter Lab. If that happens, switching to Jupyter Notebook solves the problem.

## Plotting

Plots can be done in exactly the same way as the Pandas built-in visualization tools. The only difference is that instead of typing `df.plot` we need to type `df.iplot`.

This means that to change the type of plot that will be displayed, we are using the `kind` argument.

### Line plot

```python
df.iplot()
```

### Scatter plot

By default `plotly` will try to link all the points of a graph. To prevent this behaviour and have a proper scatter plot, we need to add the `mode='markers'` argument.

```python
df.iplot(kind='scatter', x='column_A', y='column_B', mode='markers')
```

### Bar plot

```python
# If the data has already been categorised or aggregated
df.iplot(kind='bar', x='Category', y='Values')

# If the data needs to be aggregated first
# Let's take the example of a sum
df.sum().iplot(kind='bar')
```

### Box plot

```python
df.iplot(kind='box')
```

### Histogram

```python
df['column_A'].iplot(kind='hist')
```

### 3D Surface plot

Provided a data-frame with at least 3 column:

```python
df.iplot(kind='surface')
```

## Colours

Please note that the colours can be changed by using the `colorscale` argument. The documentation of which can be found [here](https://nbviewer.ipython.org/gist/santosjorge/00ca17b121fa2463e18b).

## Choropleth maps

This functionality is used to represent data over a geographical map. The documentation for `choropleth` maps can be found [here](https://plotly.com/python/choropleth-maps/). Please note that this feature does not use Cufflinks.

The documentation states that we need two types of inputs:

1. Geometry information
2. A list of values indexed by feature identifyier

This basically means that we need an object that describes the layout and a data object that is indexed by keys referring to the chosen layout. It is possible to create your own maps, but that is way to complicated, so let's just focus on the built-in ones (US-state and World Countries).

### The imports

```python
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
```

Notice that we import `chart_studio.plotly` instead of `plotly.plotly`.

### The data object

```python
data = dict(
    type='choropleth',
    locations=['AZ', 'CA', 'NY'],
    locationmode='USA-states',
    z=[1.0, 2.0, 3.0],
    text=['text 1', 'text 2', 'text 3'],
    colorscale='Portland',
    colorbar={'title': 'Colorbar Title Here'}
)
```

What does this mean ? We can see that the data is a dictionary containing the values for each geographical location.

So we start by building a dictionary and giving it a `type` key with the value `choropleth`. This is logical since that is what we want to do. Then come the **locations**.

- `locations` is a list of the locations the data is referring to.
- `locationmode` refers to the type of locations we have previously passed. In this case the states of the US. Can be `ISO-3`, `USA-states`, `country names`, `geojson-id`.
- `z` is the data we want to display per location. For example a *GDP* or *birth rates*.

Then comes the styling of the map.

- `text` is an array that will display the given text for each entry (index-based of course).
- `colorscale` defines the colours to be used (see [here](https://plotly.com/python/builtin-colorscales/))
- `colorbar` will style the legend. It takes a dictionary as argument
- `markers` will style the separation between two geographical locations (see [here](https://plotly.com/python/marker-style/))

So we can see that this data object could either be created from a pandas dataframe (where `locations`, `z` and `text`  are extracted from a specific column) or imported via an API call returning a JSON.

### The Layout object

The layout object defines what type of geographical shape we are dealing with. The official documentation explains how to create one, but I'll just use the built-ins as it is simpler:

```python
layout = dict(
    title='My wonderful title',
    geo={
        'scope': 'usa',
        'showlakes': True,
        'lakecolor': 'rgb(85, 173, 240)'
    }
)
```

Again we see that the layout object is a dictionary.

This, however defines the map itself, not the datapoints. This is why we can give the layout a title (for example: `Votes for Trump %`). Moreover we use a `geo` key. This contains the `scope` of the map as well as information regarding the color of lakes, rivers, seas etc.

### Plotting the map

```python
my_map = go.Figure(data=[data], layout=layout)
iplot(my_map)
```



---

## Errors


2.Possible Colorscale Error 2: In the "Real Data US Map Choropleth", when you are creating the data dictionary, make sure the colorscale line is = 'ylorbr', not 'YIOrbr'... so like this:

colorscale='ylorbr'

3.Possible projection Error 3: In the "World Map Choropleth", when you are creating the layout, ensure that your projection line is = {'type':'mercator'} not Mercator with a capital...so like this:

projection={'type':'mercator'}
