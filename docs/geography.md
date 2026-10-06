# Geography

Geomapviz uses caller-supplied boundaries. The installed package contains no
country files; the separate [examples bundle](examples/index.md) supplies frozen
Belgian municipalities and Dutch postcodes with complete source attribution.
Use native GeoPandas to load files and dissolve polygons.

## Boundaries and identifiers

Start with the canonical runner's local loader:

```python
--8<-- "examples/geographic_gallery.py:load"
```

`DATA` resolves beside the script. Each boundary frame needs one non-missing,
unique ID per area, an active geometry column, a declared coordinate reference
system (CRS), and valid non-empty geometry. Plotting requires polygons or
multipolygons. Repeated IDs need an explicit GeoPandas `dissolve` first.
Summary IDs must also be unique; overlapping attribute names other than the ID
fail rather than acquiring implicit suffixes.

The [postcode example](examples/netherlands.md) preserves identifiers as text.
Preparation normalizes strings and finite numeric IDs to strings, retaining
categorical labels and leading zeros. Numeric `1` and `1.0` match `"1"`, but
neither matches `"001"`. Load padded IDs as strings at ingestion; omitted zeros
cannot be inferred.

## Coordinate reference systems

A CRS relates coordinates to locations on Earth. Belgian boundaries retain
**EPSG:31370**; Dutch boundaries retain **EPSG:28992**. Both use projected metres,
so their centroids supply the simulated patterns. The
[data descriptions](examples/index.md#code-and-data) distinguish publication
year from historical boundary vintage.

Preparation preserves CRS and caller data. Static plots use the prepared CRS;
interaction transforms a copy to EPSG:3857 (Web Mercator) at rendering.
GeoPandas `to_crs` transforms coordinates; `set_crs` only labels them. Assign a
missing CRS only when you know the source system. Never relabel projected metres
as longitude and latitude.

## Coverage

`prepare_geography(summary, boundaries, geoid)` joins onto **all boundaries**.
Observation IDs without boundaries fail. Boundaries without observations retain
`NaN` metrics and support, preserving extent. Coverage metadata lists matched
and unmatched boundary IDs in boundary order, alongside summary metadata.

In [Belgian rates](examples/rates.md), Antwerpen remains visibly missing.
In [CBS percentages](examples/demographics.md), all 103 excluded postcode
boundaries remain visible. Missing coverage is not a measured zero or a reason
to remove an area from the map. Inspect each export's JSON coverage and CSV rows.
The [arithmetic reference](arithmetic.md) supplies a four-rectangle check.

## Parent areas

[Changing geographic scale](examples/aggregation.md) uses explicit Statbel
municipality-to-arrondissement/region attributes. Never infer those mappings
from ID prefixes. `assign_parent` validates one non-missing parent per base ID,
copies original records and assigns parent labels; existing labels must agree.
It preserves base IDs, row order, index and metrics.

Aggregate those original records, then dissolve geometry separately by the same
parent ID. The tutorial includes the actual canonical code. Averaging child
means or ratios ignores unequal denominators. A parent boundary can remain
missing if none of its children have records.

## Load your own boundaries

Use `geopandas.read_file` for your shapefile ZIP, GeoPackage or another supported
format. Select the ID and needed attributes, preserve CRS, and validate the
one-row-per-ID grain before preparing an existing summary. Extra attributes can
supply administrative mappings or readable names; they should not collide with
metric columns. Follow the same [independent checks](examples/index.md#numerical-checks)
for IDs, polygon validity, parent mappings and boundary coverage.
