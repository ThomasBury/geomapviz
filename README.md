# Geomapviz

**2.0 development documentation and API** — this branch is `2.0.0.dev0`.
Geomapviz prepares geographic means and exposure-weighted rate comparisons,
joins them to your boundaries, and plots observed and predicted results with
shared scales. Summaries remain inspectable pandas DataFrames.

## Installation

Use Python 3.12 by default (3.12+ is required). From this development checkout:

```sh
python -m pip install .
# For interactive maps and standalone HTML export:
python -m pip install '.[interactive]'
```

Or with uv:

```sh
uv venv --python 3.12
uv pip install .
# For interaction, use uv pip install '.[interactive]' instead.
```

Compatible wheels include precompiled native libraries. No Conda environment is
required on the verified Linux stack. See the [installation guide](docs/index.md#installation)
for platform resolution limits and optional conda-forge fallback. Installing
`geomapviz` from PyPI may still give the released 1.x API.

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

- [2.0 development guide](docs/index.md): installation and a complete synthetic comparison.
- [API and 1.x migration](docs/api.md): signatures, rendering options and breaking changes.
- [Read the Docs](https://geomapviz.readthedocs.io/en/latest/): published 1.x documentation until a development preview is activated.
- [Milestone evidence](MILESTONES.md): numerical, rendering and installation checks.

Build locally with `uv pip install --group docs` and `.venv/bin/zensical build --clean`.
The [prepared example](examples/prepared_comparison.py) exports static PNGs;
add `--interactive` for standalone HTML.
