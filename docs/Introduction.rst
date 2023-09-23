Introduction
============

Geomapviz is a Python library for visualizing geospatial tabular data. 
It aggregates tabular data at the geoid level, merges it with the shapefile, 
and provides a simple API to plot the average for single or multiple columns. 
The library is designed to create beautiful and interactive visualizations that 
help users better understand geospatial data. Geomapviz can produce a single map 
or a panel of maps, making it useful for comparing how different models capture geographical patterns. 
The package also supports returning average values either raw or automatically binned. 
Additionally, it allows users to customize the background color, 
including the option to switch from a black background to a light one. 
The styling is handled by a DataClass, PlotOptions, object is used to specify 
various arguments for creating a geospatial plot of a dataset


This library provides a simple and user-friendly interface for visualizing and analyzing spatial data. 
The library leverages a shapefile to aggregate tabular data by a geographic area and provides an intuitive 
API to plot averages for single or multiple columns. This can be particularly useful for comparing the target 
to predicted values or the aggregated predictor values on a map. With just a few lines of code, users can 
generate high-quality visualizations of their data, making it easy to draw insights and communicate their findings to others.

The plot options for the visualization of the aggregated data can be set using a dataclass provided by the library. 
This dataclass includes various parameters such as the color map, alpha, clipping colors, and more. 
By using this dataclass, the user can easily customize the appearance of the plot to their liking without 
the need to manually adjust each individual parameter.

For your convenience, the library includes shapefiles for BE and NL. However, you are free to use any 
shapefile of your choice, including official ones available on government websites or other online resources.