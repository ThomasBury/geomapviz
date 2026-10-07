# Plotting and export

`plot_geography` renders an already prepared GeoDataFrame without reaggregation
or mutation. See [Belgian means](examples/belgium.md) for continuous/classified
panels, [rate comparisons](examples/rates.md) for scales by quantity, and
[Dutch postcodes](examples/netherlands.md) for small-area interaction.

## Shared scales

Select metrics with the same units. Observed and predicted rates share a range;
metadata separates totals, ratios, differences, counts and weights. Differences
use a symmetric range centered on zero and `RdBu_r`; other quantities use
`options.cmap`. The rate tutorial shows why a comparison needs consistent colors.

Missing values do not affect limits. Constant quantity groups receive a padded
range; a constant panel otherwise shares the range of varying peers. An entirely
missing panel has no numerical colorbar.

The [administrative-scale tutorial](examples/aggregation.md) exports separate
figures with pooled limits applied through native Matplotlib normalization, and
the same limits for interaction. Independent calls normally choose their own
ranges; a region map should not silently stretch a narrower range to the full palette.

## Classification

Continuous colors are the default. `PlotOptions(autobin=True, n_bins=5)` pools
finite values across panels of each quantity into up to five Fisher–Jenks
classes, limited by distinct values. The
[Belgian classified panels](examples/belgium.md#classification) share edges and
palette; the [Dutch view](examples/netherlands.md#classified-patterns) uses the
same rules in PNG and HTML. Values on an upper edge belong to the interval
ending there. Read interval legends rather than assuming continuous values.

Differences use symmetric equal intervals with an odd number of classes,
keeping zero central. An even requested `n_bins` is reduced by one for differences.

## Missing regions and support

Missing and undefined values remain grey, outlined and hatched in static maps.
They never become zero. The rates tutorial distinguishes these meanings for
Antwerpen, Brussels and Charleroi; the CBS tutorial retains excluded boundaries.

`include_support=True` appends record count and total-weight panels discovered
from metadata, each on its own scale. The runner exports support separately so
readers can compare its patterns with rates. Hover reports ID, raw metric and
available support even without support panels. Counts and exposure describe
information volume; they are not uncertainty, confidence or prediction intervals.

## Metadata

Keep aggregation `DataFrame.attrs` through preparation and rendering. It identifies
collision-safe derived names and quantities. `prepare_geography` preserves it
and adds coverage. CSV does not retain `attrs`; the runner writes companion JSON
for inspection. Regenerate the prepared frame when reproducing maps.

Selecting/serializing columns may lose metadata or keep references to removed
columns. Keep the complete prepared frame and select panels through `metrics`.
If restoring metadata, retain every referenced support column. Without metadata,
selected metrics are assumed to share units; render different quantities separately
and omit support unless valid metadata exists.

## PNG export

Static plotting returns a Matplotlib `Figure`. Use its native `savefig` method
and close it afterwards. The canonical export helper below supplies DPI, panel
size, optional class rules and shared limits. The rates tutorial also uses
native Matplotlib for its small comparison scatterplot.

## Standalone HTML export

Install `geomapviz[interactive]==2.0.2`. Interaction returns a HoloViews layout
of GeoViews polygons. `holoviews.save(..., backend="bokeh", resources="inline")`
exports plot resources for use without a Python server or tile service.
The canonical helper retains already projected polygon data/options as native
HoloViews polygons to avoid repeated projection:

```python
--8<-- "examples/geographic_gallery.py:export"
```

[Open the Dutch postcode view](assets/geographic/netherlands_classified.html),
or use the [titled embeds and downloads](examples/netherlands.md#explore-interactively).
Full geometry makes exports large; static previews remain available. Offline
hover, pan, zoom and reset have been tested with external requests blocked.
Optional upstream stylesheet requests are unnecessary for those tools.

The package renderer starts with fixed panel dimensions at 100 pixels per inch.
The gallery helper removes fixed dimensions and uses native HoloViews sizing
with equal coordinate units. Native Bokeh auto ranges enforce the coordinate
scale as the frame resizes, including space taken by colorbars. Choose `ncols=1` when a narrow screen needs larger
individual panels. A missing extra gives an explicit installation error. Rendering preserves application Matplotlib
settings and an existing Bokeh theme. Tiles, raster backgrounds, normalization
and statistical intervals are outside the v2 API; see [PlotOptions](api.md#plotoptions).
