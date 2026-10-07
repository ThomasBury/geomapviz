# Geomapviz

[![Belgian municipality boundaries colored by a simulated weighted geographic signal.](docs/assets/geographic/belgium_mean.png)](docs/examples/belgium.md)

Geomapviz calculates geographic means and exposure-weighted rate comparisons
from tabular records, joins them to your boundaries, and draws maps with shared
color scales. Summaries remain inspectable pandas DataFrames. This map uses
real Belgian municipality boundaries and simulated values.

From a checkout, run the working Belgian demonstration:

```sh
uv run --no-project --python 3.12 --with 'geomapviz==2.0.2' \
  examples/geographic_gallery.py --case belgium --output output
```

Open `output/belgium_mean.png`. The [walkthrough](docs/examples/belgium.md)
explains equal/weighted means, four related signals and classification.
The script reads local snapshots; running needs no data downloads, tiles or server.

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
