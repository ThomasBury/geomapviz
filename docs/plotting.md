# Plotting and export

**2.0 development documentation.** `plot_geography` renders an already prepared
GeoDataFrame. It does not aggregate records or change caller data.

## Shared scales

Select metrics that measure the same quantity. Observed and predicted rates
share a scale across panels. Metadata gives totals, ratios, differences, counts
and weights separate scales. Differences use a symmetric scale centered on
zero and the `RdBu_r` diverging palette. Other quantities use `options.cmap`.

When all finite values in a quantity group are equal, its shared range is padded.
A constant panel otherwise shares the range of its varying peers. Missing values
do not influence scale limits; an entirely missing panel has no numerical colorbar.

## Classification

Continuous colors are the default. `PlotOptions(autobin=True, n_bins=7)` pools
finite values across panels of the same quantity and uses at most seven
Fisher–Jenks classes, limited by the number of distinct values. The class edges
and palette are shared across the static and interactive backends.

Differences instead use symmetric equal intervals with an odd number of
classes, keeping zero in the central class. An even requested `n_bins` is
reduced by one for differences. Class legends show intervals; values on an
upper edge belong to the interval ending at that edge.

```python
# Continue from the Quickstart's mapped frame and metrics.
from geomapviz.plot import PlotOptions, plot_geography

classified = plot_geography(
    mapped, metrics, geoid="area", include_support=True,
    options=PlotOptions(autobin=True, n_bins=5, ncols=3),
)
classified.savefig("classified.png")
plt.close(classified)
```

## Missing regions and support

Missing and undefined values are grey with an outline, and hatched in static
maps. They never become zero. The example's `004` is missing in every panel;
`003` is zero in the loss panel and undefined in the model A ratio panel.

`include_support=True` appends count and total-weight panels from summary
metadata. Each has its own scale. Interactive hover includes the geographic
ID, raw metric and available support even when support panels are not requested.
Support indicates the records and exposure behind a result; it is not uncertainty.

## Metadata

Keep `DataFrame.attrs` from aggregation through preparation and rendering.
The renderer uses it to discover collision-safe derived names and distinguish
quantities. `prepare_geography` preserves it and adds coverage.

CSV does not preserve `attrs`. Selecting or serializing columns can also lose
metadata or retain references to columns you removed. Keep the complete prepared
frame and use the renderer's `metrics` argument to select panels. If you restore
metadata yourself, also retain every referenced support column.

Without metadata, selected metrics are assumed to share units. Plot different
quantities in separate calls, and omit `include_support=True` unless valid
support metadata is present.

## PNG export

Static plotting returns a Matplotlib `Figure`. Use its native export method:

```python
import matplotlib.pyplot as plt

figure = plot_geography(
    mapped, metrics, geoid="area", include_support=True,
    options=PlotOptions(ncols=3, figsize=(15, 10), alpha=0.8),
)
figure.savefig("comparison.png", dpi=150)
plt.close(figure)
```

## Standalone HTML export

Install `.[interactive]` from the development checkout. Interactive plotting
returns a HoloViews layout of GeoViews polygons. Inline resources make the
export usable offline with no Python server or map-tile downloads:

```python
import holoviews as hv

layout = plot_geography(
    mapped, metrics, geoid="area", include_support=True,
    options=PlotOptions(interactive=True, ncols=3, figsize=(15, 10), alpha=0.8),
)
hv.save(layout, "comparison.html", backend="bokeh", resources="inline")
```

[Open or download the interactive example](assets/comparison.html).
The exported layout has fixed panel dimensions: use `ncols=1` for a narrow
screen and choose an appropriate `figsize` before export. Interactive dimensions
use 100 pixels per inch. Missing the extra gives an explicit installation error.

The renderers preserve application Matplotlib settings and an existing Bokeh
theme. Tiles, raster backgrounds, normalization and statistical intervals are
outside the 2.0 API. See [PlotOptions](api.md#plotoptions) for all controls.
