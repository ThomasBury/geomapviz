# Geomapviz

**Geomapviz 2.0.0** — Python 3.12+ is required.
Geomapviz prepares geographic means and exposure-weighted rate comparisons,
joins them to your boundaries, and plots observed and predicted results with
shared scales. Summaries remain inspectable pandas DataFrames.

## Installation

Use Python 3.12 by default (3.12+ is required):

```sh
python -m pip install geomapviz==2.0.0
# For interactive maps and standalone HTML export:
python -m pip install 'geomapviz[interactive]==2.0.0'
```

Or with uv:

```sh
uv venv --python 3.12
uv pip install geomapviz==2.0.0
# For interaction, use uv pip install 'geomapviz[interactive]==2.0.0' instead.
```

Compatible wheels include precompiled native libraries. No Conda environment is
required on the verified Linux stack. See the [installation guide](docs/index.md#installation)
for platform resolution limits and optional conda-forge fallback. For the 1.x
API, install `geomapviz==1.1.3` and use its documentation.

## Example

Supply your own boundaries; this example invents two rectangles. Observed loss
is an amount, while `model` is a predicted rate per unit of exposure.

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
summary = aggregate_rates(records, "area", "loss", ["model"], "exposure")
mapped = prepare_geography(summary, boundaries, "area")
figure = plot_geography(mapped, ["loss", "model"], geoid="area", include_support=True)
figure.savefig("comparison.png")
plt.close(figure)
```

Area `001` has observed rate 10 and predicted rate 11; area `002` remains missing.
No boundaries, rasters or sample datasets are bundled.

## Documentation

- [2.0 guide](docs/index.md): installation and a complete synthetic comparison.
- [API and 1.x migration](docs/api.md): signatures, rendering options and breaking changes.
- [2.0.0 documentation](https://geomapviz.readthedocs.io/en/v2.0.0/).
- [Release notes](docs/release-notes.md): breaking changes and validation limits.
- [1.x documentation](https://geomapviz.readthedocs.io/en/1.1.3/).
- [Milestone evidence](MILESTONES.md): numerical, rendering and installation checks.

Build locally with `uv pip install --group docs` and `.venv/bin/zensical build --clean`.
The [prepared example](examples/prepared_comparison.py) exports static PNGs;
add `--interactive` for standalone HTML.
