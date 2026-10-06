# Geography

**Geomapviz 2.0.0 documentation.** Geomapviz uses caller-supplied boundaries.
No country files, raster backgrounds or sample datasets are bundled.

## Boundaries and identifiers

Continue in the same Python session as the [Quickstart](index.md#a-complete-synthetic-comparison),
using its synthetic records, summary and four boundaries.
The boundary frame must have one non-missing, unique ID per area, an active
geometry column, a declared coordinate reference system (CRS), and valid,
non-empty geometries. Plotting requires polygons or multipolygons.

Duplicate boundary IDs require an explicit GeoPandas `dissolve` first.
Summary IDs must also be unique. Overlapping column names other than the ID
raise an error rather than silently adding suffixes.

Geography preparation normalizes strings and finite numeric IDs to strings.
Categorical labels and leading zeros survive: `"001"` remains `"001"`.
Numeric `1` and `1.0` match `"1"`, but neither matches `"001"`.
Load padded identifiers as strings at ingestion; missing zeros cannot be inferred.

## Coordinate reference systems

A CRS describes how coordinates relate to locations on Earth. Preparation
preserves the boundary CRS and does not change your input frame. Static plots
use the prepared CRS. The interactive renderer transforms a copy to EPSG:3857
(Web Mercator) at the rendering boundary.

Use GeoPandas `to_crs` if you want another static projection. `set_crs` labels
existing coordinates; it does not transform them. Assign a missing CRS only
when you know the source coordinate system.

## Coverage

`prepare_geography` joins summaries onto all boundaries. Any observation ID
without a boundary raises an error. Boundaries without observations remain
in the output with `NaN` metrics and support, preserving the geographic extent.

```python
# Continue from the Quickstart's summary and boundaries.
mapped = prepare_geography(summary, boundaries, "area")
assert mapped.attrs["coverage"] == {
    "matched": ["001", "002", "003"],
    "unmatched": ["004"],
}
assert mapped.loc[mapped["area"] == "004", "loss"].isna().all()
assert mapped.crs == boundaries.crs
```

Coverage metadata lists matched and unmatched boundary IDs in boundary order.
An unmatched boundary is missing information, not a zero observed rate.
Summary metadata is retained alongside coverage.

## Parent areas

`assign_parent` copies the original records and assigns a parent ID using the
validated base-boundary mapping. It requires one non-missing parent per base ID;
if records already have parent labels, they must agree. Base IDs, row order,
index and metrics remain intact.

Aggregate the assigned original records, then dissolve the boundary geometries
separately. Never average child-area means or ratios: their denominators differ.
Using the Quickstart data:

```python
from geomapviz import aggregate_rates, assign_parent, prepare_geography

parent_boundaries = boundaries.assign(region=["north", "north", "south", "empty"])
parent_records = assign_parent(records, parent_boundaries, "area", "region")
parent_summary = aggregate_rates(
    parent_records, "region", "loss", ["model_a", "model_b"], "exposure"
)
regions = parent_boundaries[["region", parent_boundaries.geometry.name]].dissolve(
    by="region", as_index=False
)
parent_map = prepare_geography(parent_summary, regions, "region")
north = parent_summary.set_index("region").loc["north"]
assert north["loss_total"] == 48
assert north["total_weight"] == 6
assert north["loss"] == 8
assert north["model_a"] == 9
```

The north observed rate is `48 / 6 = 8`, whereas averaging the child rates
`10` and `4` would incorrectly give `7`. The `empty` parent boundary remains
missing. See [Aggregation](aggregation.md) for denominator and support rules.

## Load your own boundaries

Load a boundary file with `geopandas.read_file`, choose its geographic ID column,
and pass only needed boundary attributes to `prepare_geography`.
With an existing aggregate `summary` whose area IDs occur in the file:

```python
import geopandas as gpd
from geomapviz import prepare_geography

file_boundaries = gpd.read_file("boundaries.gpkg")
file_mapped = prepare_geography(
    summary, file_boundaries[["area", file_boundaries.geometry.name]], "area"
)
```
