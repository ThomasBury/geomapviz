<img src="pics/logo.png" alt="drawing" width="200"/>

[buy me caffeine](https://ko-fi.com/V7V72SOHX)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/geomapviz?style=flat)](https://pypi.org/project/geomapviz/)

# 🗺️🐍 Geomapviz - Python Library for Beautiful and Interactive Geospatial Tabular Data Visualization 🚀

Geomapviz is a Python library for visualizing geospatial tabular data. It aggregates tabular data at the geoid level, merges it with the shapefile, and provides a simple API to plot the average for single or multiple columns. The library is designed to create beautiful and interactive visualizations that help users better understand geospatial data. Geomapviz can produce a single map or a panel of maps, making it useful for comparing how different models capture geographical patterns. The package also supports returning average values either raw or automatically binned. Additionally, it allows users to customize the background color, including the option to switch from a black background to a light one. The styling is handled by a DataClass, PlotOptions, object is used to specify various arguments for creating a geospatial plot of a dataset

[Geomapviz ReadTheDocs](https://geomapviz.readthedocs.io/en/latest/)

<td align="left"><img src="pics/example_01.png" width="600"/></td>
<td align="left"><img src="pics/example_02.png" width="300"/></td>


## Installation

The 2.0 development checkout requires Python 3.12 or newer. Install the base
package for aggregation, geography preparation and static maps:

```sh
pip install .
```

For interactive maps and HTML export:

```sh
pip install '.[interactive]'
```

Country boundaries, raster backgrounds and sample data are no longer bundled.
Supply your own boundaries with `geopandas.read_file`, then use
`aggregate_means` or `aggregate_rates`, `prepare_geography`, and
`geomapviz.plot.plot_geography`. The obsolete `load_shp`, `load_geometry`,
`merge_zip_df` and `convert_category_to_code` helpers have been removed.
The hosted documentation below still describes the 1.x API.

## Documentation

The [documentation notebook](nb/docs/geomap.ipynb) illustrates the functionality of `geomapviz`
## Changelog

### 1.0

 - Complete refactoring of the library, including modular features and simpler code base

### 0.6

 - Including files in source distributions

### 0.5

 - [Bug] Capital letter in importing the BE shapefile
 - [Bug] Changed default values of arguments

### 0.4

 - Make Belgian shp available using load_be_shp
 - More decimal
 - User defined alpha for the interactive maps

### 0.3

 - Bound functions to the upper level

### 0.2

 - First version

### 0.1

 - First version
