# Geographic examples

Real Belgian municipalities and Dutch postcodes make spatial comparisons visible.
Four tutorials use simulated values; the fifth uses measured CBS resident counts.
All run through one ordinary Python script with local, committed data.

| Walkthrough | What to look for |
| --- | --- |
| [Belgium: geographic means](belgium.md) | Equal and weighted means; four comparable signals; continuous colors and classes |
| [Belgium: changing geographic scale](aggregation.md) | Municipality, arrondissement and region patterns; original-record aggregation and separate dissolve |
| [Belgium: observed and predicted rates](rates.md) | Directional errors; shared rates, differences, ratios, support and scatterplots |
| [Netherlands: postcode geography](netherlands.md) | Countrywide continuous and classified patterns; small-area interaction |
| [Netherlands: measured population data](demographics.md) | Age percentages from published counts; denominators, rounding and missing coverage |

## Code and data

[Download the reproduction ZIP](../downloads/geographic-examples.zip).
It contains the canonical `geographic_gallery.py`, the independent numerical
checker, instructions, both complete shapefile ZIPs, population CSV, provenance,
checksums, licences and snapshot preprocessing code. The geographic files are
outside package installations. No notebook server or application is needed.

Code blocks in these tutorials are included directly from named sections of the
canonical script using [Zensical source snippets](https://github.com/zensical/docs/blob/master/docs/authoring/code-blocks.md).
They are excerpts; run the full downloaded script to supply imports and shared
helpers. The bundle's `data/README.md` describes sources, historical boundary
vintages and field mappings; `data/manifest.json` records exact checksums.

## Run all cases

Extract the ZIP, open a terminal in its directory, and run:

```sh
uv run --no-project --python 3.12 --with 'geomapviz==2.0.1' \
  geographic_gallery.py --output output

uv run --no-project --python 3.12 --with 'geomapviz[interactive]==2.0.1' \
  geographic_gallery.py --case all --interactive --output output
```

[uv supplies Python and dependencies](https://docs.astral.sh/uv/guides/scripts/).
`--no-project` avoids installing the surrounding checkout. Dependency installation
needs internet access initially; running examples reads only local files and
uses no data downloads, tiles or Python server. The default is all static cases;
`--case` selects `belgium`, `aggregation`, `rates`, `netherlands` or `demographics`.
`--output` selects a directory; data paths always resolve beside the script.

Open PNGs in an image viewer and standalone HTML directly in a browser. HTML
contains full geometry and inline plot resources, so files are large (5–93 MB)
and rendering can take time. Published embeds are optional and keep static
previews available. Hover, pan, zoom and reset work with external requests
blocked; optional upstream stylesheet requests are not needed for those tools.

## Inspect the results

Every case writes summary CSVs and JSON with geographic IDs, CRS, support names
and boundary coverage. Synthetic cases also write input records (aggregation
uses the same inputs as `belgium`). CSV does not retain pandas metadata; keep
JSON or regenerate the prepared frame before rendering quantity comparisons.
Synthetic patterns use projected-coordinate centroids, seed 2024 and positive
weights. Statistical support counts and exposure do not estimate uncertainty.

## Numerical checks

```sh
uv run --no-project --python 3.12 --with 'geomapviz==2.0.1' \
  check_geographic_gallery.py --exports output
```

The checker validates snapshot checksums, unique IDs, CRS, polygon validity,
parent mappings and coverage. It independently calculates equal/weighted means,
rates and parent results from original records and verifies exported summaries.
It distinguishes a missing municipality, a genuine zero and an undefined ratio,
and checks the common valid CBS cohort. Rectangles remain useful for
[compact arithmetic checks](../arithmetic.md).
