# Geomapviz

Geomapviz helps Python data analysts summarize records by area and map averages
or actual-versus-predicted rates on their own geographic boundaries.

## TL;DR

[Gallery](https://geomapviz.readthedocs.io/en/stable/examples/)

- Calculate averages by area, with optional weights.
- Group smaller areas into larger districts or regions using your area mapping.
- Match results to your own map boundaries.
- Compare actual and predicted rates, including differences and ratios.
- Automatically group values into color ranges, shared across comparable maps shown together.
- Create static images or interactive HTML maps, and inspect the underlying pandas tables.

## When is it useful?

| Situation | How Geomapviz helps |
| --- | --- |
| Your table contains many records for each town or postcode. | [Summarize them into one result per area](docs/examples/belgium.md). Use weights when records represent different populations or amounts of activity. |
| You need both local detail and a regional overview. | [Group geographic IDs using your area mapping](docs/examples/aggregation.md) and recalculate results from the original records. |
| You want to find where predictions are too high or too low. | [Compare actual and predicted rates](docs/examples/rates.md) on matching scales, then map their differences and ratios. |
| Small color differences make a map difficult to read. | [Use automatic bins](docs/examples/belgium.md#classification) to show values in a few color ranges, shared across comparable maps shown together. |

**Aggregation** means summarizing rows by area. **Geographic IDs** are the codes
identifying those areas, such as town codes or postcodes; they connect your
records to your boundaries.

To move from towns → districts → regions, supply a mapping of smaller-area IDs
to larger-area IDs. Geomapviz assigns those IDs with `assign_parent`; then use
`aggregate_means` or `aggregate_rates` to recalculate summaries from the original
records and weights. GeoPandas merges the boundaries with `dissolve`.

**Autobinning** chooses numerical color ranges automatically, so nearby values
can share a color. Enable it with `autobin=True` and set the maximum number of
ranges with `n_bins`; the [illustrated example](docs/examples/belgium.md#classification)
shows five ranges.

Calculate a summary once, inspect its pandas table, and reuse it for static or
interactive maps. Comparable maps shown together share colors for the same
values; rates, differences and ratios use separate scales. Areas without data
remain visible as missing, distinct from measured zero.

## Try the Belgian example

[![Belgian municipality boundaries colored by a simulated weighted geographic signal.](docs/assets/geographic/belgium_mean.png)](docs/examples/belgium.md)

This map uses real Belgian municipality boundaries and **simulated values**.

From a checkout, run the working Belgian demonstration:

```sh
uv run --no-project --python 3.12 --with 'geomapviz==2.0.2' \
  examples/geographic_gallery.py --case belgium --output output
```

Open `output/belgium_mean.png`. The [walkthrough](docs/examples/belgium.md)
explains equal/weighted means, four related signals and classification.
The script reads local snapshots; running needs no data downloads, tiles or server.

### From a DataFrame to one value per area

Here is a smaller teaching DataFrame using two IDs from the same Belgian
boundary snapshot. These three simulated records make the arithmetic easy to
follow; the map above uses the full demonstration dataset.

```python
import geopandas as gpd
import pandas as pd

from geomapviz import aggregate_means, assign_parent, prepare_geography

records = pd.DataFrame({
    "mun_id": ["11001", "11001", "11002"],
    "simulated_signal": [10.0, 20.0, 30.0],
    "exposure": [1.0, 3.0, 2.0],
})
boundaries = gpd.read_file("examples/data/belgium_municipalities_2024.zip")
```

`records` contains several rows per municipality:

| mun_id | simulated_signal | exposure |
| --- | ---: | ---: |
| 11001 | 10.0 | 1.0 |
| 11001 | 20.0 | 3.0 |
| 11002 | 30.0 | 2.0 |

```python
summary = aggregate_means(records, "mun_id", ["simulated_signal"], weight="exposure")
mapped = prepare_geography(summary, boundaries, "mun_id")
```

Geomapviz groups the records, calculates the weighted means and adds support
columns automatically. The resulting `summary` is still a pandas DataFrame:

| mun_id | simulated_signal | support_count | total_weight |
| --- | ---: | ---: | ---: |
| 11001 | 17.5 | 2 | 4.0 |
| 11002 | 30.0 | 1 | 2.0 |

For municipality `11001`, the mean is `(10 × 1 + 20 × 3) / (1 + 3) = 17.5`.
`prepare_geography` validates the IDs and joins these results to the polygons,
preserving the boundary coordinate system. Municipalities without records stay
on the map with missing values. Pass `mapped` to `plot_geography` to render it.

### Group smaller geographic IDs into larger areas

The supplied boundaries map both `11001` (Aartselaar) and `11002` (Antwerpen)
to arrondissement `11000`. Geomapviz uses that mapping to assign the parent ID
to every original record and calculate a new weighted mean:

```python
parent_records = assign_parent(records, boundaries, "mun_id", "arr_id")
parent_summary = aggregate_means(
    parent_records, "arr_id", ["simulated_signal"], weight="exposure"
)

# GeoPandas merges all municipality polygons belonging to each arrondissement.
parent_boundaries = boundaries[["arr_id", "geometry"]].dissolve(by="arr_id").reset_index()
parent_mapped = prepare_geography(parent_summary, parent_boundaries, "arr_id")
```

The resulting `parent_summary`, rounded here to two decimals, is:

| arr_id | simulated_signal | support_count | total_weight |
| --- | ---: | ---: | ---: |
| 11000 | 21.67 | 3 | 6.0 |

Its mean is `(10 × 1 + 20 × 3 + 30 × 2) / (1 + 3 + 2) = 21.67`, rather than
the unweighted average of the two municipality means, `23.75`. The polygons
merge separately from the statistical calculation. Use `reg_id` instead of
`arr_id` to group by region; the [scale walkthrough](docs/examples/aggregation.md)
shows all three geographic levels for the full dataset.

## Installation

[uv](https://docs.astral.sh/uv/guides/scripts/) supplies Python and dependencies
for the demonstration. `--no-project` avoids installing checkout dependencies.
For an existing Python 3.12+ environment:

```sh
python -m pip install geomapviz==2.0.2
# Add interactive maps and standalone HTML:
python -m pip install 'geomapviz[interactive]==2.0.2'
```

The geographic data is in the separate examples, outside installed package
files. See the [Quickstart](docs/quickstart.md) for environment setup and the
[gallery](docs/examples/index.md) for the downloadable reproduction bundle.

## Examples

[![Dutch postcode maps of measured percentages of residents under 15 and aged 65 or older.](docs/assets/geographic/demographics.png)](docs/examples/demographics.md)

- [Belgium: geographic means](docs/examples/belgium.md)
- [Belgium: changing geographic scale](docs/examples/aggregation.md)
- [Belgium: observed and predicted rates](docs/examples/rates.md)
- [Netherlands: postcode geography](docs/examples/netherlands.md)
- [Netherlands: measured population data](docs/examples/demographics.md)

Each illustrated Markdown tutorial uses code from the canonical Python runner,
describes its data and includes reproduction commands. Synthetic demonstrations
are labelled; the population tutorial uses published CBS counts. Full geometry,
licences and provenance are committed under `examples/data/`.

## Documentation and project information

- [Published 2.0.2 documentation](https://geomapviz.readthedocs.io/en/v2.0.2/)
- [Aggregation](docs/aggregation.md), [geography](docs/geography.md) and [plotting](docs/plotting.md) guides
- [API and 1.x migration](docs/api.md)
- [Release notes](docs/release-notes.md)
- [Archived 1.1.3 tutorials](https://geomapviz.readthedocs.io/en/1.1.3/nb/geomap.html)
- [Report an issue](https://github.com/ThomasBury/geomapviz/issues)
- [MIT license](LICENSE.md); geographic data has its own included licences
- [Build the documentation](docs/quickstart.md#build-the-documentation)
