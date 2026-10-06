# Release notes

## 2.0.0

Geomapviz now prepares inspectable geographic means and exposure-weighted
observed/model rate comparisons before rendering. This is a breaking release;
there are no 1.x compatibility shims. Python 3.12 or newer is required.

### Breaking changes

- Replace `prepare_dataframe`, `compute_weighted_average` and
  `weighted_average_aggregator` with `aggregate_means` or `aggregate_rates`.
  Results are pandas DataFrames with original metric names, support and
  collision-safe derived columns, rather than renamed `target` metrics or tuples.
- Observed inputs must explicitly represent totals or rates; predictions are
  rates. Differences are observed minus predicted rates. Observed/expected
  ratios use compatible totals and remain missing when expected totals are zero.
- Missing/non-finite metrics, invalid weights, empty input and groups without
  positive support now fail explicitly, including invalid zero-weight records.
  Select the common cohort before aggregation; valid zero-weight records do not
  contribute to totals, means or support counts.
- Replace `spatial_average_plot` and `spatial_average_facetplot` with
  `prepare_geography` followed by `plot_geography`. `PlotOptions` contains only
  rendering settings. Unknown/duplicate IDs, invalid geometries, missing CRS and
  conflicting columns fail explicitly. Unobserved boundaries remain missing.
- Derive parent labels with `assign_parent`, aggregate original records, and
  dissolve boundaries with GeoPandas. Parent rates are recomputed from totals
  and exposure rather than averaged from child rates.
- Removed `compute_confidence_interval`, interval outputs, `distr`, uncertainty,
  tiles, raster backgrounds and normalization options. Support is not uncertainty.
- Removed `load_shp`, `load_geometry`, `merge_zip_df`,
  `convert_category_to_code` and all bundled boundaries, rasters and sample data.
  Supply your own files and read them with native GeoPandas/pandas operations.
- Interactive dependencies are optional. Install
  `geomapviz[interactive]==2.0.0` for HoloViews/GeoViews rendering and standalone
  HTML; the base package supports preparation and static maps.
- The contributor `doc` extra is replaced by the `docs` dependency group for
  the Zensical guide. Runtime dependencies use tested floors; see
  [milestone evidence](https://github.com/ThomasBury/geomapviz/blob/v2.0.0/MILESTONES.md).

### Rendering and migration

Original rates share scales; totals, ratios, signed differences, counts and
exposure use separate scales. Differences use symmetric limits around zero.
Missing areas are distinct from true zeros. PNG and inline HTML use native
Matplotlib and HoloViews exports; no map tiles or Python server are required.

Keep summary `attrs` through preparation and rendering: it records derived
column names, support and quantity types. CSV does not preserve this metadata.
See [API and migration](api.md#migrating-from-1x) and the [Quickstart](index.md).
For the previous API, use `geomapviz==1.1.3` and the
[1.x documentation](https://geomapviz.readthedocs.io/en/1.1.3/).

### Verification limits

Runtime and distribution checks cover standard CPython 3.12 and 3.14 on Linux
x86_64, with current dependencies and a tested older stack on Python 3.12.
Wheel availability was checked for macOS and Windows; those systems were not
runtime-tested. Synthetic examples establish arithmetic, coverage and rendering
behavior, not production adoption or statistical calibration. Existing
mapclassify fallback and older-stack upstream deprecation warnings remain.
