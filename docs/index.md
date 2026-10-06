# Geomapviz

[![Recognizable Belgian municipality boundaries colored by a simulated geographic signal.](assets/geographic/belgium_mean.png)](examples/belgium.md)

Geomapviz turns tabular records into geographic means and exposure-weighted
rate comparisons, joins them to your boundaries, and draws comparable maps.
The summaries remain inspectable pandas DataFrames. These are real Belgian
boundaries; their values are simulated.

Download and extract the [geographic examples](downloads/geographic-examples.zip),
then run this working Belgian demonstration from the extracted directory:

```sh
uv run --no-project --python 3.12 --with 'geomapviz==2.0.0' \
  geographic_gallery.py --case belgium --output output
```

Open `output/belgium_mean.png`. No data downloads, tiles or Python server are
needed when running the script. [The walkthrough](examples/belgium.md) explains
the signal, weighting and four comparable maps.

## Installation

[uv](https://docs.astral.sh/uv/getting-started/installation/) supplies Python and
dependencies for the command above. `--no-project` avoids installing a surrounding
checkout's dependencies. For an existing Python 3.12+ environment:

```sh
python -m pip install geomapviz==2.0.0
# For interactive maps and standalone HTML:
python -m pip install 'geomapviz[interactive]==2.0.0'
```

See the [Quickstart](quickstart.md#installation) for environment setup and platforms.

## Explore geographic examples

[![Dutch postcode boundaries colored by measured percentages of residents under 15 and aged 65 or older.](assets/geographic/demographics.png)](examples/demographics.md)

Compare [Belgian means](examples/belgium.md),
[changing administrative scale](examples/aggregation.md) and
[observed versus predicted rates](examples/rates.md). Explore
[Dutch postcode patterns](examples/netherlands.md) and
[measured CBS population counts](examples/demographics.md).
Each tutorial includes results, source code, data notes and a reproduction command.

Start with the [Quickstart](quickstart.md), or consult the
[guides](aggregation.md) and [API and migration](api.md).
The archived [1.1.3 tutorials](https://geomapviz.readthedocs.io/en/1.1.3/nb/geomap.html)
and [2.0.0 documentation](https://geomapviz.readthedocs.io/en/v2.0.0/)
remain available. Large geographic data files are in the separate examples
bundle, outside package installations.
