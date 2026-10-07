# Quickstart

Start with the [Belgian municipality example](examples/belgium.md), using real
boundaries and simulated records. Download and extract the
[reproduction bundle](downloads/geographic-examples.zip), then run from the
extracted directory:

```sh
uv run --no-project --python 3.12 --with 'geomapviz==2.0.2' \
  geographic_gallery.py --case belgium --output output
```

Open `output/belgium_mean.png`. It shows an exposure-weighted simulated signal
for 581 municipalities; `belgium_predictors.png` compares four related signals.
The CSVs let you inspect the records and equal/weighted means behind the maps.

## Installation

Use Python 3.12 by default; the package requires Python 3.12 or newer.
Create an environment and install the release from PyPI:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install geomapviz==2.0.2
```

On Windows, create the environment with `py -3.12 -m venv .venv` and use
`.venv\Scripts\python.exe` wherever this guide uses `.venv/bin/python`.
The base installation supports aggregation, geography preparation and static maps.
For interactive maps and standalone HTML export, install the extra in the same environment:

```sh
.venv/bin/python -m pip install 'geomapviz[interactive]==2.0.2'
```

With [uv](https://docs.astral.sh/uv/getting-started/installation/), the equivalent is:

```sh
uv venv --python 3.12
uv pip install geomapviz==2.0.2
# Choose this instead for HTML export:
uv pip install 'geomapviz[interactive]==2.0.2'
```

For the previous API, install `geomapviz==1.1.3` and use the
[1.x documentation](https://geomapviz.readthedocs.io/en/1.1.3/).

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

## Load, aggregate, prepare, plot

The canonical runner loads a local zipped shapefile with GeoPandas:

```python
--8<-- "examples/geographic_gallery.py:load"
```

`DATA` is the directory beside the script, resolved with
`Path(__file__).resolve().parent / "data"`. The files contain complete CRS
metadata; Belgium uses projected metres in EPSG:31370.

The following function is included directly from the runner. `simulate` creates
reproducible records; `export` calls `plot_geography` and writes PNGs, plus HTML
when requested. Run the complete script above to supply these helpers:

```python
--8<-- "examples/geographic_gallery.py:belgium"
```

This separates three steps: aggregate records into an inspectable pandas
summary, join it to boundaries, then render the prepared geography.
For a fully printed small calculation, use the [arithmetic reference](arithmetic.md).

## Next steps

Work through the [five geographic examples](examples/index.md), then use the
[aggregation](aggregation.md), [geography](geography.md) and
[plotting](plotting.md) guides for validation and rendering details.

## Build the documentation

From a checkout, install only the existing documentation group. Build the
untracked bundle before the site; the site copies committed previews and does
not execute the gallery:

```sh
uv venv --python 3.12
uv pip install --group docs
mkdir -p docs/downloads
(cd examples && ../.venv/bin/python -m zipfile -c ../docs/downloads/geographic-examples.zip \
  geographic_gallery.py check_geographic_gallery.py GEOGRAPHIC_EXAMPLES.md \
  data)
.venv/bin/zensical build --clean
.venv/bin/python examples/check_documentation.py
.venv/bin/zensical serve
```

Zensical uses its current default theme. Read the bundle's `GEOGRAPHIC_EXAMPLES.md`
for reproduction and `data/README.md` for provenance. To refresh previews, run
`examples/geographic_gallery.py --case all --interactive --output output` in an
interactive installation, inspect the exports, then copy PNGs and the selected HTML views to
`docs/assets/geographic/`. Normalize trailing line whitespace in generated HTML
before committing. Full geometry makes HTML larger and slower to render.
