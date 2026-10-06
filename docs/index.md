# Quickstart

**2.0 development documentation** — this checkout is `2.0.0.dev0`.
The published 1.x package has a different API; see [API and migration](api.md#migrating-from-1x).

Geomapviz prepares geographic means and exposure-weighted rate comparisons,
joins them to your boundaries, and renders maps with shared scales. Use it when
you need to compare observed outcomes with several models on the same records.
The summaries remain ordinary pandas DataFrames that you can inspect before plotting.

## Installation

Use Python 3.12 by default; the package requires Python 3.12 or newer.
Run these commands from a checkout of the `feat/v2.0.0` branch:

```sh
git clone --branch feat/v2.0.0 https://github.com/ThomasBury/geomapviz.git
cd geomapviz
python3.12 -m venv .venv
.venv/bin/python -m pip install .
```

On Windows, create the environment with `py -3.12 -m venv .venv` and use
`.venv\Scripts\python.exe` wherever this guide uses `.venv/bin/python`.
The base installation supports aggregation, geography preparation and static maps.
For interactive maps and standalone HTML export, install the extra in the same environment:

```sh
.venv/bin/python -m pip install '.[interactive]'
```

With [uv](https://docs.astral.sh/uv/getting-started/installation/), the equivalent is:

```sh
uv venv --python 3.12
uv pip install .
# Choose this instead for HTML export:
uv pip install '.[interactive]'
```

These install the development checkout. `pip install geomapviz` installs the
published release, which may still be 1.x.

Compatible wheels contain precompiled native libraries, so a normal pip or uv
installation does not require you to compile those libraries.
[GeoPandas installation guidance](https://geopandas.org/en/stable/getting_started/install.html)
describes its binary dependencies;
[Cartopy installation guidance](https://cartopy.readthedocs.io/stable/installing.html)
documents wheels since 0.22. Wheel-only resolution of the interactive stack passed
for Linux x86_64 with glibc 2.28+, macOS Intel 13+, Apple Silicon macOS 13 with
Python 3.12 (macOS 14+ with Python 3.14), and Windows x86_64. These checks establish
dependency availability; actual installation and rendering were verified on Linux.

If your platform has no compatible wheels, conda-forge is an optional fallback.
[Pixi](https://pixi.prefix.dev/latest/switching_from/uv/) is useful when you need
to manage native libraries or tools through conda-forge. This project provides
no Pixi environment or additional lockfile.

## A complete synthetic comparison

This is the same data and workflow as
[the prepared-comparison example](https://github.com/ThomasBury/geomapviz/blob/feat/v2.0.0/examples/prepared_comparison.py).
The four rectangles are invented boundaries, not an administrative map.
`loss` contains observed amounts; both model columns contain predicted amounts
per unit of exposure. Exposure measures the time or quantity at risk, such as
insured years. With loss in euros and exposure in insured years, the resulting
rates are euros per insured year. The zero-exposure row has no contribution.

`area` is the geographic ID that joins records to boundaries; keep its leading
zeros. `EPSG:4326` identifies the boundary coordinate reference system (CRS):
the rectangles use longitude and latitude in degrees.

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box

from geomapviz import aggregate_rates, prepare_geography
from geomapviz.plot import PlotOptions, plot_geography

records = pd.DataFrame({
    "area": pd.Categorical(["001", "001", "002", "002", "003"]),
    "loss": [10.0, 30.0, 8.0, 999.0, 0.0],
    "exposure": [1.0, 3.0, 2.0, 0.0, 1.0],
    "model_a": [8.0, 12.0, 5.0, 999.0, 0.0],
    "model_b": [10.0, 10.0, 3.0, 999.0, 1.0],
})
boundaries = gpd.GeoDataFrame(
    {"area": ["001", "002", "003", "004"]},
    geometry=[box(4 + i * 0.1, 50, 4.08 + i * 0.1, 50.08) for i in range(4)],
    crs="EPSG:4326",
)
summary = aggregate_rates(
    records, "area", "loss", ["model_a", "model_b"], "exposure"
)
mapped = prepare_geography(summary, boundaries, "area")
metrics = ["loss", "model_a", "model_b", "model_a_difference", "model_a_ratio"]
figure = plot_geography(
    mapped, metrics, geoid="area", include_support=True,
    options=PlotOptions(ncols=3, figsize=(15, 10), alpha=0.8),
)
figure.savefig("comparison.png")
plt.close(figure)
print(summary)
print(mapped.attrs["coverage"])
```

Save the code as `quickstart.py` in the checkout directory and run:

```sh
.venv/bin/python -i quickstart.py
```

Open `comparison.png` from that directory in an image viewer. The `-i` option
keeps the Python session open; paste continuation examples from the following
pages into that same session to reuse `records`, `boundaries`, `summary` and `mapped`.

Area `001` has observed rate `40 / 4 = 10` and model A rate
`(8 × 1 + 12 × 3) / 4 = 11`. Its signed difference is `-1` and ratio is
`40 / 44`. A negative observed-minus-predicted difference or an observed/expected
ratio below one indicates overprediction. Area `003` has a real zero observed
rate and an undefined model A ratio because the expected total is zero. Area
`004` has no observations and retains missing values.
[Aggregation](aggregation.md) explains these distinctions.

[![Seven synthetic maps show observed and two predicted rates, their signed difference and ratio, record counts and exposure. Area 004 is hatched as missing; area 003 has zero loss and an undefined model A ratio.](assets/comparison.png)](assets/comparison.png)

Select the image to open the full-resolution PNG. In each panel, areas run
`001`–`004` from left to right.

[Open or download the standalone interactive comparison](assets/comparison.html).
Hover reveals IDs, raw values and support; pan and zoom use the map toolbar.
The HTML contains its JavaScript resources and needs no Python server.

## Next steps

- [Aggregation](aggregation.md): select a common cohort and distinguish totals from rates.
- [Geography](geography.md): validate coverage and aggregate parent areas correctly.
- [Plotting and export](plotting.md): classification, support panels and export commands.
- [API and migration](api.md): signatures, options and the 1.x replacements.

## Build this guide

Contributors can build or preview without installing Geomapviz or its geospatial dependencies:

```sh
uv venv --python 3.12
uv pip install --group docs
.venv/bin/zensical build --clean
.venv/bin/zensical serve
```

Zensical 0.0.68 uses the default theme. The build copies the committed PNG and
HTML; it does not execute examples. To regenerate them, use an interactive
installation and run:

```sh
.venv/bin/python examples/prepared_comparison.py --interactive --output /tmp/geomapviz-doc-assets
```

Copy `continuous.png` and `continuous.html` from that directory to
`docs/assets/comparison.png` and `docs/assets/comparison.html`. Strip trailing
line whitespace from the HTML before committing, then rebuild.
