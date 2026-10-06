# Geomapviz

Geomapviz calculates geographic averages from tabular records and maps them onto
your boundaries. It compares observed and predicted rates, weighted by exposure,
using shared color scales so you can compare areas and models.
Summaries remain inspectable pandas DataFrames before plotting.

- [Documentation / Quickstart](https://geomapviz.readthedocs.io/en/v2.0.0/)
- [API reference](https://geomapviz.readthedocs.io/en/v2.0.0/api/)
- [1.x migration guide](https://geomapviz.readthedocs.io/en/v2.0.0/api/#migrating-from-1x)

## Installation

Geomapviz 2.0.0 requires Python 3.12 or newer. Install it in your Python environment.

Base package for static maps:

```sh
python -m pip install geomapviz==2.0.0
```

Interactive extra for interactive maps and standalone HTML export:

```sh
python -m pip install 'geomapviz[interactive]==2.0.0'
```

See the [installation guide](https://geomapviz.readthedocs.io/en/v2.0.0/#installation)
for uv commands, environment setup and platform considerations.

## Your first map

Supply tabular records and polygon boundaries with matching geographic IDs.
Boundaries need a declared coordinate reference system (CRS), which describes how
coordinates relate to locations on Earth. This example invents two rectangles
in longitude and latitude (`EPSG:4326`).

Exposure is the time or quantity at risk, such as insured years. Here, `loss`
contains observed amounts and `model` contains predicted rates per unit of
exposure. With loss in euros and exposure in insured years, both output rates
are euros per insured year.

Save this as `first_map.py` and run `python first_map.py`:

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box

from geomapviz import aggregate_rates, prepare_geography
from geomapviz.plot import plot_geography

records = pd.DataFrame({
    "area": ["001", "001"],
    "loss": [10.0, 30.0],
    "model": [8.0, 12.0],
    "exposure": [1.0, 3.0],
})
boundaries = gpd.GeoDataFrame(
    {"area": ["001", "002"]},
    geometry=[box(4, 50, 4.08, 50.08), box(4.1, 50, 4.18, 50.08)],
    crs="EPSG:4326",
)
# 1. Aggregate records into rates by area.
summary = aggregate_rates(records, "area", "loss", ["model"], "exposure")
# 2. Join the summary to boundaries.
mapped = prepare_geography(summary, boundaries, "area")
# 3. Plot observed and predicted rates on a shared color scale.
figure = plot_geography(mapped, ["loss", "model"], geoid="area")
figure.savefig("comparison.png")
plt.close(figure)
```

Area `001` has observed rate `40 / 4 = 10` and predicted rate
`(8 × 1 + 12 × 3) / 4 = 11`. Area `002` has no records and remains missing.
Open `comparison.png` in your working directory to see the two panels.

The [complete tutorial](https://geomapviz.readthedocs.io/en/v2.0.0/#a-complete-synthetic-comparison)
adds models, differences and ratios; see its
[map preview](https://geomapviz.readthedocs.io/en/v2.0.0/assets/comparison.png).

## Learn more

- [Aggregate means and rates](https://geomapviz.readthedocs.io/en/v2.0.0/aggregation/).
- [Prepare boundaries and check coverage](https://geomapviz.readthedocs.io/en/v2.0.0/geography/).
- [Plot maps and export PNG or HTML](https://geomapviz.readthedocs.io/en/v2.0.0/plotting/).

## Project information

- [Release notes](https://geomapviz.readthedocs.io/en/v2.0.0/release-notes/)
- [Archived 1.x documentation](https://geomapviz.readthedocs.io/en/1.1.3/)
- [Report an issue](https://github.com/ThomasBury/geomapviz/issues)
- [MIT license](LICENSE.md)
- [Build the documentation](https://geomapviz.readthedocs.io/en/v2.0.0/#build-this-guide)
