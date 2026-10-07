# Run the geographic gallery

Keep `geographic_gallery.py`, `check_geographic_gallery.py` and the `data/`
directory together. You may run them from any working directory. Data paths
are resolved relative to the script, and `--output` selects the export directory.

After installing [uv](https://docs.astral.sh/uv/getting-started/installation/),
uv supplies Python 3.12 and the requested package dependencies automatically:

```sh
uv run --no-project --python 3.12 --with 'geomapviz==2.0.2' \
  geographic_gallery.py --case belgium --output output

uv run --no-project --python 3.12 --with 'geomapviz[interactive]==2.0.2' \
  geographic_gallery.py --case rates --interactive --output output
```

`--no-project` prevents uv from installing a surrounding checkout's project
dependencies. See [uv's script guide](https://docs.astral.sh/uv/guides/scripts/).
Dependency installation needs internet access initially; the examples themselves
make no requests, use no map tiles, and need no Python server. Open the exported
standalone HTML directly in a browser.

The runner defaults to `--case all` and all static exports. Cases:

| Case | Results and what to look for |
| --- | --- |
| `belgium` | Municipality mean, four comparable simulated signals/predictors, continuous and classified maps. Compare local noise with east/north biases; equal and exposure-weighted summaries are exported separately. |
| `aggregation` | Municipality, arrondissement and region maps on one continuous scale. Regional aggregation smooths variation. Original records are reaggregated; geometries are dissolved separately using authoritative mappings. |
| `rates` | Simulated observed counts per exposure and two simulated rate predictions, shared scales, signed differences, ratios, counts, exposure and a Matplotlib scatterplot. Models have different east/north spatial errors. |
| `netherlands` | Four countrywide continuous postcode maps of projected-coordinate simulated patterns and one classified signal map. Add `--interactive` to inspect small areas. |
| `demographics` | Measured CBS percentages under 15 and aged 65 or older, with a shared scale and excluded boundaries visible. The CSV shows the common valid cohort. |

Every case exports PNGs, inspectable CSV records/summaries and JSON containing
CRS, support names and boundary coverage. `--interactive` adds inline standalone
HTML maps. Synthetic cases use a fixed seed, projected-coordinate centroids,
positive exposure weights and unequal record counts. Simulated means use
`sum(value * exposure) / sum(exposure)`; equal means give every record weight one.
Simulated observed rates use `sum(observed_count) / sum(exposure)`; model columns
contain rates and are exposure-weighted. Support counts and exposure describe
how much information was aggregated; they are not uncertainty estimates.
Support counts count input records: in the CBS example there is one input row
per valid postcode, while its total weight is the published resident denominator.

In `rates`, Antwerpen (`11002`) is deliberately missing, Bruxelles/Brussel
(`21004`) has a genuine observed zero, and Charleroi (`52011`) has zero predicted
rate for the east model, making its observed/expected ratio undefined. Grey
means missing or undefined, never zero. The CSVs distinguish these cases.

Classification pools values across comparable panels into five Fisher–Jenks
classes; difference scales are symmetric around zero. All four simulated signal
columns have the same units. Rate, difference, ratio, count and exposure maps
use separate scales. Static and interactive outputs use the same class rules.

Source dates, redistribution licences, field mappings, suppression rules and
checksums are in [data/README.md](data/README.md) and [data/manifest.json](data/manifest.json).
The data describes historical boundaries and 2024 population, not present-day
administrative geography. Full geometry is retained, so HTML exports are larger
than the PNGs.

Run the independent snapshot and numerical checks with:

```sh
uv run --no-project --python 3.12 --with 'geomapviz==2.0.2' \
  check_geographic_gallery.py
```
