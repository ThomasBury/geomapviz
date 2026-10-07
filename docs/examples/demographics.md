# Netherlands: measured population data

These values are **measured public CBS counts**, not the simulations in the other
tutorials. Map the percentages of registered residents under 15 and aged 65 or
older using the same 2024 postcode boundaries and a common valid cohort.

[![Two Dutch postcode maps showing percentages of residents under 15 and aged 65 or older, with excluded areas grey.](../assets/geographic/demographics.png)](../assets/geographic/demographics.png)

Both panels use one continuous percentage scale. Look for areas where the two
age profiles differ, and for grey areas where the published counts do not support
a calculation. A percentage describes composition, not the number of residents;
a large colored polygon need not represent many people.

## Counts, denominators and rounding

The companion CSV keeps `residents`, `under15` and `age65plus`, with one row per
postcode and the reference year 2024. The denominator is all registered residents
on 1 January, for both age groups:

- `under15_percent = 100 * under15 / residents`
- `age65plus_percent = 100 * age65plus / residents`

CBS rounds counts to multiples of five. These percentages are approximate ratios
of the rounded counts, rather than CBS's separately published rounded percentage
fields. Rounding can matter especially when denominators are small. These data
describe registered residents, not everyone physically present in the area.

## Suppression and the valid cohort

CBS code `-99997` means 0–4 / suppressed / absent; `-99995` means not yet published.
Neither is a measured zero. The source CSV preserves these codes. The runner
masks negative counts before calculation, then requires all three counts and a
positive resident denominator in one explicit cohort for both panels.

This gives **3,968 valid postcodes** and **103 excluded postcodes**. All excluded
boundaries remain visible with missing metrics. `demographics_cohort.csv` records
cohort membership and masked counts; `demographics.csv` contains the mapped
percentages, including missing rows. Grey is missing coverage, not zero percent.

## Canonical code

```python
--8<-- "examples/geographic_gallery.py:demographics"
```

`aggregate_means` receives one percentage row per valid postcode and a resident
weight. At this scale, its mean equals the input percentage. `support_count` is
therefore one per included postcode; `total_weight` is its published resident
count. If regrouping these rows, a resident-weighted percentage recovers
`100 * sum(age_count) / sum(residents)` for the selected cohort. Do not average
postcode percentages equally or treat support as a measure of uncertainty.

## Data and attribution

Source: **CBS / PDOK, postcode4 statistics 2024**, retrieved 6 October 2026 from
the [PDOK collection](https://api.pdok.nl/cbs/postcode4/ogc/v1?f=html&lang=en),
filtered to `jaarcode=2024`. The frozen EPSG:28992 boundaries contain 4,071 unique
text postcode IDs. The bundle includes selected published counts as CSV and
source/derived checksums, field mappings and preprocessing instructions.

CBS's [publication notes, sections 4.3 and 4.6](https://www.cbs.nl/nl-nl/longread/diversen/2025/statistische-gegevens-per-vierkant-en-postcode-2022-2023-2024/4-beschrijving-cijfers)
describe the count definitions, rounding and suppression. Counts and geometry
are redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
with selection/conversion by the Geomapviz documentation contributors. The
licence text accompanies the data; CBS and PDOK do not endorse these tutorials.

## Explore interactively

[Open the full-page map](../assets/geographic/demographics.html) ·
<a href="../assets/geographic/demographics.html" download>Download standalone HTML (39 MB)</a>.
Use hover for the ID, value and support; use the toolbar to pan, zoom and reset.
The PNG above remains available without loading the interactive view.

<details ontoggle="if (this.open) { const frame = this.querySelector('iframe'); if (!frame.hasAttribute('src')) frame.src = frame.dataset.src; }">
<summary>Open the interactive map (39 MB)</summary>
<iframe data-src="../assets/geographic/demographics.html" title="Interactive percentages of Dutch registered residents under 15 and aged 65 or older" width="100%" height="500" loading="lazy"></iframe>
</details>

## Reproduce

[Download the bundle](../downloads/geographic-examples.zip), extract it, and run
from its directory. [uv](https://docs.astral.sh/uv/guides/scripts/) supplies Python
and dependencies; `--no-project` avoids installing checkout dependencies.

```sh
uv run --no-project --python 3.12 --with 'geomapviz==2.0.2' \
  geographic_gallery.py --case demographics --output output

uv run --no-project --python 3.12 --with 'geomapviz[interactive]==2.0.2' \
  geographic_gallery.py --case demographics --interactive --output output
```

The first command exports PNGs, inspectable CSVs and metadata JSON; the second
also exports standalone HTML. Data paths are relative to the script, so running
needs no data downloads, map tiles or Python server. Full geometry makes HTML
exports large and slower to generate.
