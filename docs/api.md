# API and migration

**Geomapviz 2.0.0 documentation.** Python 3.12+ is required.
This reference is hand-authored;
the [Quickstart](index.md) provides a runnable comparison.

## Preparation functions

Import these from `geomapviz`. Inputs are not mutated.

### aggregate_means

```python
aggregate_means(df, geoid, metrics, weight=None)
```

`df` is a pandas DataFrame; `geoid` and optional `weight` are column names;
`metrics` is a non-empty list of unique metric column names. Returns a DataFrame
of named equal-weight or weighted means with support metadata.
See [means and validation](aggregation.md#means).

### aggregate_rates

```python
aggregate_rates(df, geoid, observed, predicted, exposure, *, observed_kind="total")
```

`observed`, `exposure` and `geoid` are column names. `predicted` is a non-empty
list of unique predicted-rate column names. `observed_kind` accepts `"total"`
or `"rate"`. Returns aggregate rates under the original metric names, totals,
signed observed-minus-predicted differences, observed/expected ratios and
support. `attrs` maps derived names and records the observed input kind.
See [rate semantics](aggregation.md#observed-totals-and-rates).

### prepare_geography

```python
prepare_geography(summary, boundaries, geoid)
```

`summary` has one row per observed area; `boundaries` is a GeoDataFrame with
one unique geographic ID per boundary. Returns a GeoDataFrame retaining every
boundary, its CRS, summary metadata and `attrs["coverage"]`.
Unknown summary IDs, duplicate IDs, invalid geometry, missing CRS and attribute
collisions raise errors. See [Geography](geography.md).

### assign_parent

```python
assign_parent(records, boundaries, geoid, parent)
```

`geoid` is the base ID column and `parent` is a distinct parent ID column on
validated boundaries. Returns copied records with parent labels assigned from
the boundary mapping; conflicting existing labels raise an error.
It does not aggregate records or dissolve geometries.
See [parent areas](geography.md#parent-areas).

## plot_geography

Import from `geomapviz.plot`:

```python
plot_geography(mapped, metrics, *, geoid, include_support=False, options=None)
```

`mapped` is a prepared GeoDataFrame of polygons or multipolygons; `metrics` is
a non-empty list of unique numeric column names. `geoid` is required.
`include_support=True` appends support columns and requires their metadata.
`options=None` uses default `PlotOptions`.

Returns a Matplotlib `Figure`, or a HoloViews `Layout` when interaction is
requested. One metric gives one map; incomplete facet grids are supported.
No reaggregation occurs. See [plotting and exports](plotting.md).

## PlotOptions

Import `PlotOptions` from `geomapviz.plot`. This dataclass contains rendering
controls only, with no records, geometry or cohort rules.

| Field | Default | Meaning |
| --- | --- | --- |
| `figsize` | `(12, 8)` | Two finite positive dimensions in inches; interaction uses 100 pixels/inch |
| `ncols` | `2` | Positive integer number of columns, limited to the panel count |
| `cmap` | `"viridis"` | Matplotlib colormap for quantities other than differences |
| `facecolor` | `"white"` | Matplotlib-compatible background color |
| `alpha` | `1.0` | Finite opacity between 0 and 1 |
| `autobin` | `False` | Enable shared classification by quantity |
| `n_bins` | `7` | Positive integer requested class count |
| `interactive` | `False` | Return a HoloViews layout; requires the interactive extra |

Options are validated at construction and again before rendering, so field
edits cannot bypass validation. See [classification](plotting.md#classification)
for differences and class-count limits.

## Migrating from 1.x

2.0 is a breaking release with no compatibility shims. Released Git tags retain
the original sources; [published 1.x documentation](https://geomapviz.readthedocs.io/en/1.1.3/)
remains separate from this guide. See the [2.0.0 release notes](release-notes.md).

| Removed 1.x behavior | 2.0 replacement |
| --- | --- |
| `prepare_dataframe`, weighted-average helpers and renamed `target` metrics | `aggregate_means` preserves metric names and returns a DataFrame |
| Implicit weighting and confidence intervals | `aggregate_rates` distinguishes observed totals/rates; predictions are rates; intervals are removed |
| Country loaders, sample bundles and CSV/category helpers | Read caller-supplied boundaries with GeoPandas; preserve geographic labels |
| `spatial_average_plot`, `spatial_average_facetplot` and data-bearing `PlotOptions` | Prepare with `prepare_geography`, then render with `plot_geography` |
| Record-supplied parent mapping and averages of area averages | `assign_parent`, aggregate original records, and dissolve boundaries explicitly |
| Tiles, raster backgrounds, normalization and uncertainty options | Shared scales, missing-area styling and explicit support panels |
| Interactive stack in every installation | Install the `interactive` extra when needed |

The obsolete `load_shp`, `load_geometry`, `merge_zip_df` and
`convert_category_to_code` helpers are removed. Use native GeoPandas and pandas
file readers instead.

Select a common finite cohort, supply valid weights/exposure, and check coverage
before plotting. Zero-support observed areas and unknown IDs fail explicitly;
unobserved boundaries remain missing. Preserve summary metadata for support
names and quantity-specific scales; CSV does not retain it.
