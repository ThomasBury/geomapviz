# Geomapviz 2.0 implementation evidence

## M0 — complete (2026-10-02)

Created local branch `feat/v2.0.0` from `main` at
`3d1d45e31809135f7cab8d76e1e38838385ba1b2` before package changes.
The base version is `1.1.3`; main and the 1.x history remain available.
The only initial working-tree change was untracked `PRD.md`, preserved as supplied.
No push or publication is part of this work.

Environment: Linux x86_64, glibc 2.39, CPython 3.12.7, uv 0.12.15.
Installed `.[test,lint]` and hvPlot in a local `.venv` for the native baseline.
Resolved versions: pandas 3.0.6, NumPy 2.5.3, GeoPandas 1.2.0,
Matplotlib 3.11.2, HoloViews 1.23.2, GeoViews 1.15.1, hvPlot 0.12.2.

Baseline results, before repairs:

- `python -m pytest -q`: collection fails because `geomapviz.geomapviz` is absent.
- `python -m black --check src tests`: existing plot and shapefile modules fail.
- `python -m flake8 src tests`: existing long lines, invalid docstring escapes,
  and an unused pytest import fail. No repository flake8 configuration exists.
- The weighted mean of `[10, 20]` with weights `[1, 3]` is 17.5, but the
  metric is renamed to `target`. `[10, missing]` with the same weights is
  silently accepted and returns 2.5.
- `import geomapviz.plot` succeeds with the complete installed stack.

These are baseline defects, not regressions from M2. Rendering repairs,
dependency cleanup, and distribution verification belong to M3–M6.

## M1 — complete: continue with exposure/rate comparison

Maintainer direction: the user confirmed the exposure/rate workflow on
2026-10-02. Continue 2.0 for a validated numerical and geographic comparison
contract, not for generic choropleth styling. No anonymized production case was
provided. Synthetic feasibility and the requested workflow support this scoped
decision; they do not establish adoption or broad demand.

Runnable native baseline:

```sh
uv pip install --python .venv/bin/python hvplot 'cartopy==0.25.0'
.venv/bin/python examples/native_comparison.py
```

Five synthetic records, three observed areas, and four EPSG:4326 polygons test
unequal exposure, a zero-exposure record, two models, leading-zero categorical
IDs, zero expected loss, and an unobserved boundary. Hand-worked checks pass:
area 001 has observed total 40, exposure 4, observed rate 10, model A expected
total 44 and rate 11, difference -1, and observed/expected ratio 10/11. Area
003 has an undefined ratio for model A. Input data is unchanged. Coverage is
three matched boundaries and one unmatched boundary, whose metric stays missing.

Native GeoPandas/pandas and hvPlot/GeoViews export the same prepared summary to
`/tmp/geomapviz-m1/native-static.png` and `native-interactive.html`. Static
panels use verified shared limits [0, 11]; the unobserved area is hatched grey,
distinct from a true zero. Interactive panels use the same limits and expose
IDs, counts, and exposure on hover. The PNG was visually inspected.

| Criterion | Native baseline finding | Minimum package benefit |
| --- | --- | --- |
| Statistical correctness | Caller constructs a positive-exposure cohort, sums observed totals once, weights predicted rates, and handles zero expected totals | One checked total/rate contract reused across analyses and models |
| Coverage | Caller checks missing/duplicate boundary IDs, geometry/CRS and unknown observation IDs, then left joins | Consistent join validation and inspectable coverage in M3 |
| Comparison quality | Native shared limits and hover information already work | Keep preparation inspectable; retain only small rendering functions in M4 |
| Repeated effort | Each caller repeats cohort validation, support, rate arithmetic, differences/ratios, and join checks | Two pandas summary entry points, without intervals or renamed metrics |
| Installation cost | Runtime resolution: 61 packages; static native baseline: 17; interactive native baseline: 46 | Optional interactive dependencies and asset removal in M5; no reduction claimed yet |
| Maintenance | Existing historical runtime assets total 148,975,966 bytes; rendering compatibility still needs ownership | No boundary downloads, imagery, new backends, or new uncertainty methods |

Counts come from Python 3.12 `uv pip compile` of the project runtime, of
`geopandas + matplotlib`, and of `geopandas + hvplot + geoviews`; development
tools are excluded. The installed baseline with development tools has 79
distributions and measured 863,573,563 bytes before the Cartopy diagnostic.
These are a local snapshot, not installation-speed or savings claims.

The initially resolved Cartopy 0.26.0 with GeoViews 1.15.1 fails during export
with `KeyError: lon_0`; the explicit EPSG:4326/PlateCarree baseline exports with
Cartopy 0.25.0. No core dependency bounds were changed: M6 must revisit this
compatibility issue before release. Plotly was not evaluated because the user
selected the existing exposure/rate workflow and supplied no Plotly requirement.

Retain in M2: arbitrary named weighted/equal-weight means; observed totals or
existing observed rates with exposure; predicted rates; signed observed-minus-
predicted rate differences; observed-to-expected total ratios; positive-support
counts and exposure; ordinary pandas results with collision-safe derived names.
Keep M3–M6 pending. The package must justify its continued maintenance through
this shared contract; native rendering alone would not justify it.

Capability references refreshed with Context7 and official sources on 2026-10-02:
[GeoPandas aggregation](https://geopandas.org/en/stable/docs/user_guide/aggregation_with_dissolve.html),
[GeoPandas mapping](https://github.com/geopandas/geopandas/blob/main/doc/source/docs/user_guide/mapping.rst),
[hvPlot geographic data](https://hvplot.holoviz.org/en/docs/latest/user_guide/Geographic_Data.html),
[hvPlot export](https://github.com/holoviz/hvplot/blob/main/doc/user_guide/Viewing.ipynb).
GeoPandas dissolve defaults to the first attribute value; totals/weights must
be summed explicitly before deriving parent-area rates, rather than averaging
already aggregated rates.

## M2 — complete: numerical contract

Development version is `2.0.0.dev0`, with Python minimum 3.12. The setuptools
backend and dynamic version source are retained. Public numerical operations
import only pandas/NumPy and return ordinary DataFrames:

```python
import pandas as pd
from geomapviz import aggregate_means, aggregate_rates

records = pd.DataFrame({
    "area": ["001", "001"],
    "loss": [10.0, 30.0],
    "prediction": [8.0, 12.0],
    "exposure": [1.0, 3.0],
})
comparison = aggregate_rates(
    records, geoid="area", observed="loss", predicted=["prediction"],
    exposure="exposure", observed_kind="total",
)
assert comparison.loc[0, "loss"] == 10.0       # 40 / 4
assert comparison.loc[0, "prediction"] == 11.0  # (8 * 1 + 12 * 3) / 4
assert comparison.loc[0, "loss_total"] == 40.0
assert comparison.loc[0, "prediction_total"] == 44.0
assert comparison.loc[0, "prediction_difference"] == -1.0

means = aggregate_means(records, "area", ["prediction"], weight="exposure")
assert means.loc[0, "prediction"] == 11.0
```

`aggregate_means` uses equal weights when `weight=None`. `aggregate_rates`
defaults to observed totals; set `observed_kind="rate"` for existing observed
rates. Prediction columns always contain rates. Results retain the original
metric names for aggregate means/rates and expose totals separately, preventing
double exposure weighting. Differences are observed minus predicted rates;
ratios divide compatible observed/expected totals and are NaN for zero expected.

Both operations reject empty input, missing IDs, missing/non-finite metrics,
negative/non-finite weights, duplicate column/metric names, and groups with no
positive weight. Invalid zero-weight records are still rejected: filtering the
common cohort is the caller's explicit responsibility. Valid zero-weight rows
contribute to neither means, totals, nor support counts. Overflow is rejected
instead of returning infinite summaries. IDs retain their labels and dtype,
including categorical labels, leading zeros, and numeric identifiers. Only
observed categories produce groups, and inputs are unchanged.

Support columns default to `support_count` and `total_weight`; derived total,
difference and ratio columns use the names shown above. On collision, underscores
are appended until unique. Inspect `summary.attrs["support"]`,
`comparison.attrs["totals"]`, and `comparison.attrs["comparisons"]` for the
actual names. These mappings are DataFrame metadata; formats such as CSV do not
persist them, so capture them before exporting when names collide.

Breaking changes: removed `prepare_dataframe`, categorical encoding in
aggregation, `compute_weighted_average`, `weighted_average_aggregator`, and
`compute_confidence_interval`. There is no tuple/interval output, `distr`, or
`PlotOptions.plot_uncertainty`. Use the two public operations above rather than
the renamed `target` column. No compatibility shims were added.

Existing mean-rendering callers now consume the new numerical results and use
the actual observed metric label. GeoPandas loads only inside the legacy geometry
helper. A small synthetic rendering check exercises both single and facet maps,
including metrics literally named `avg` and `count`, and verifies the prepared
17.5 mean and absence of interval columns. This caller adaptation does not
complete M3/M4: legacy geometry coverage/parent mappings, interactive CRS handling,
layout options, shared scales, opacity and import-style side effects still need
their planned work. The legacy renderer temporarily reserves ID names `model`,
`avg`, `count`, and `weight`; the numerical API has no such restriction. The
unrelated legacy CSV mapping helper remains unchanged for later cleanup.

Verification:

- CPython 3.12.7 with the full baseline stack: `python -m pytest -q` — 22 pass.
- CPython 3.14.7 with only NumPy 2.5.3, pandas 3.0.6 and pytest 9.1.1:
  `python -m pytest -q tests/test_aggregation.py` — 20 pass. Installed the project
  editable with `--no-deps`; Matplotlib, GeoPandas, HoloViews and GeoViews are absent.
  This verifies numerical independence, not the base installation promised by M5.
- A fresh subprocess verifies importing and using both public functions does not
  load any rendering stack. Numerical values match the independent M1 baseline.
- Black with target Python 3.12 passes on changed source, tests and the M1 example.
  Flake8 passes on those files with `--extend-ignore E501`, retaining the project's
  Black line-length convention. No lint configuration or tool migration was added.
- M1 synthetic static/interactive exports and assertions still pass. The known
  Cartopy 0.26/GeoViews incompatibility remains recorded above.
- `git diff --check` passes. Whole-repository formatting/lint failures in untouched
  legacy files remain baseline work, not hidden by this focused check.

M0–M2 are delivered as separate local Conventional Commits. PRD.md remains the
user-supplied untracked file; M3–M6, publication, CI/CD and site documentation
remain pending.

## Tooling follow-up — Ruff complete, ty pending (2026-10-05)

Replaced Black and Flake8 in the `lint` extra with Ruff. Ruff handles both
linting and formatting; this migration is not a prerequisite for M3–M6.
Renamed the Black workflow to `.github/workflows/ruff.yml`, set Python 3.12,
and replaced its Black step with `ruff check src tests` and
`ruff format --check src tests`. Updated the README badge and ignored Ruff's
cache. Explicit lint rules are `E4,E7,E9,F`, with the existing 88-character
formatting convention; the formatter handles wrapping without enforcing
Flake8's 79-character limit. The only source change is a formatter-added blank
line in `shapefiles.py`. The existing coverage setup is unchanged and has not
been verified on GitHub.

Verification with Ruff 0.16.10 installed in the project's `.venv`:

- `.venv/bin/ruff check src tests` passes.
- `.venv/bin/ruff format --check src tests` passes: seven files formatted.
- `.venv/bin/python -m pytest -q` passes: 22 tests on CPython 3.12.7.
- `git diff --check` passes.

Add ty separately after the legacy renderer cleanup in M4. It adds type checking,
rather than replacing linting or formatting. Check `src` against the project's
installed dependencies and fix remaining type contracts without blanket
suppressions. Require `ty check src` and the existing pytest suite to pass.

Initial read-only review evidence (2026-10-02), using Ruff 0.16.2 and ty 0.0.71
from existing local executables, with ty resolving dependencies from this
project's `.venv`:

- `ruff check --select E4,E7,E9,F src tests` passes.
- `ruff format --check src tests` would reformat only `shapefiles.py`; six files
  are already formatted.
- Existing Flake8 7.4.1 reports 91 violations, all `E501` line-length complaints.
- `ty check src --python .venv` reports 18 diagnostics; 17 concern legacy plotting
  and shapefile modules. Defer their type cleanup until the retained code is clear.

No tooling migration or type repairs were applied during that initial review.
Ruff is now migrated; ty and type repairs remain deferred until after M4.

## M3 — complete: geography, coverage and parent areas (2026-10-05)

Two public operations in the existing geometry module use user-supplied
boundaries. `prepare_geography(summary, boundaries, geoid)` returns an ordinary
GeoDataFrame in the boundary CRS, with every boundary retained. Summary metadata
survives, and `mapped.attrs["coverage"]` lists `matched` and `unmatched` boundary
IDs in boundary order. Unmatched metrics and support remain missing, distinct
from observed zeros. Caller data and geometry are unchanged.

Both keys normalize to strings without stripping, padding or guessing labels.
Categorical labels and leading zeros survive. Numeric `1`, numeric `1.0` and
text `"1"` match; text `"001"` remains a different ID. Missing/non-finite IDs,
duplicate boundary IDs (including collisions after normalization), duplicate
summary IDs, unknown observation IDs, missing columns/CRS, and missing, empty
or invalid geometries raise explicit errors. A boundary must have one active
geometry column. Overlapping boundary/summary column names fail rather than
silently suffixing or overwriting metrics; select the desired boundary columns
before joining.

`assign_parent(records, boundaries, geoid, parent)` assigns parent IDs from one
validated boundary per base ID. Parent labels must be present on all boundaries;
existing labels in records must agree. Record order, index (including duplicate
index labels), original base IDs and metrics are preserved. Reuse the M2
operations on these records so totals, exposures and support are recomputed from
the common cohort. Geometry dissolution uses native GeoPandas separately:

```python
from geomapviz import aggregate_rates, assign_parent, prepare_geography

parent_records = assign_parent(records, boundaries, "area", "region")
summary = aggregate_rates(
    parent_records, "region", "loss", ["model_a", "model_b"], "exposure",
)
parent_boundaries = (
    boundaries[["region", boundaries.geometry.name]]
    .dissolve(by="region", observed=True)
    .reset_index()
)
mapped = prepare_geography(summary, parent_boundaries, "region")
coverage = mapped.attrs["coverage"]
```

For the synthetic north region, observed total is 48, exposure is 6, observed
rate is 8 and model A expected total is 54 with rate 9. The difference is -1,
ratio is 8/9 and support count is 3. Averaging the two base-area rates would
incorrectly yield 7. Both observed-total and observed-rate inputs have checks.
An unobserved parent boundary remains present with missing metrics.

Both legacy mean-rendering callers now route through the same geography
validation. Parent mode derives its mapping from boundaries instead of trusting
record labels. Every unmatched boundary has a row for each metric in the
temporary long-format result. `load_geometry` now preserves the file CRS instead
of forcing Web Mercator. The shared interactive polygon helper transforms its
coordinates with `to_crs(epsg=3857)` before declaring Web Mercator. Coordinate
and per-polygon round-trip checks cover EPSG:4326 and Belgian Lambert EPSG:31370.

Geometry, Cartopy and historical resource imports are deferred so the new public
exports preserve M2's numerical import independence. Historical loaders/assets,
plotting options/styles, and dependency declarations remain M4/M5 work. No new
dependencies, compatibility shims, geometry repairs or guessed CRS were added.

Verification with CPython 3.12.7, pandas 3.0.6, NumPy 2.5.3, GeoPandas 1.2.0,
Shapely 2.1.2 and the existing full rendering stack:

- `.venv/bin/python -m pytest -q` — 43 pass, including all M2 checks and 21
  synthetic geography/projection cases; no historical country files are used.
- `.venv/bin/ruff check src tests` and
  `.venv/bin/ruff format --check src tests` pass.
- M1 native arithmetic, coverage and static/interactive export assertions pass;
  outputs are in `/tmp/geomapviz-m3/native-baseline`. The existing Cartopy 0.25.0
  baseline remains in use; M6 still owns the recorded newer-version compatibility
  check and fresh installations.
- `git diff --check` passes.

M3 is complete. M4 is the next implementation milestone; ty remains deferred
until its legacy-renderer cleanup. Existing Ruff migration changes and the
user-supplied untracked PRD are preserved separately. No push or publication.

## M4 — complete: prepared comparison rendering (2026-10-05)

`geomapviz.plot.plot_geography(mapped, metrics, geoid=..., options=...)` now
renders the ordinary M3 GeoDataFrame directly. `PlotOptions` contains rendering
settings only: `figsize`, `ncols`, `cmap`, `facecolor`, `alpha`, `autobin`,
`n_bins`, and the typed constructor flag `interactive`. It returns a native
Matplotlib Figure or a HoloViews Layout of GeoViews Polygons. Data preparation
remains explicit and numerical summaries remain usable with native libraries:

```python
from geomapviz import aggregate_rates, prepare_geography
from geomapviz.plot import PlotOptions, plot_geography

summary = aggregate_rates(records, "area", "loss", ["model_a", "model_b"], "exposure")
mapped = prepare_geography(summary, boundaries, "area")
figure = plot_geography(
    mapped, ["loss", "model_a", "model_b", "model_a_difference", "model_a_ratio"],
    geoid="area", include_support=True,
    options=PlotOptions(ncols=3, alpha=0.8),
)
figure.savefig("comparison.png")

import holoviews as hv

layout = plot_geography(
    mapped, ["loss", "model_a", "model_b"], geoid="area", include_support=True,
    options=PlotOptions(interactive=True, ncols=1),
)
hv.save(layout, "comparison.html", backend="bokeh", resources="inline")
```

Original metrics share limits pooled from all selected finite values, without
percentile clipping or treating missing areas as zero. Summary metadata assigns
separate scales to totals, ratios, signed differences, support counts and
weight/exposure. Difference scales are symmetric around zero with a diverging
palette. `include_support=True` adds both actual support columns, each on its
own scale. Interactive hover includes the original ID and metric labels plus
support, including collision-adjusted names. Native dimension aliases prevent
coordinate/color-field and sanitized-name collisions, while preserving the raw
columns in the returned object's data.

With `autobin=True`, compatible panels share pooled Fisher-Jenks classes,
capped by the number of distinct finite values. Inputs are rescaled to [0, 1]
for classification to avoid variance cancellation on tightly spaced values;
boundaries come back from the original data. Signed differences instead use
symmetric equal intervals with an odd class count no greater than `n_bins`,
keeping zero in the middle class. Both backends use identical class assignments
and palettes, and retain raw values for hover/inspection. Legend precision
increases when needed to distinguish close boundaries. Constant data and fewer
distinct values than requested classes render without inventing missing zeros.
Entirely missing panels show their geometry without a numerical colorbar.

Missing/undefined regions are grey and outlined, with hatching and a legend on
static maps and a title note interactively. Each polygon is drawn once. Opacity
is respected on both backends; white and dark backgrounds have readable titles
and legends. Single maps, one-column comparisons and incomplete grids work.
Coordinates transform to EPSG:3857 before the interactive Web Mercator
projection; the M3 EPSG:4326/Belgian Lambert round-trip checks still pass.

Breaking changes: removed `spatial_average_plot`, `spatial_average_facetplot`,
the temporary `dissolve_and_aggregate` long-format adapter and plotting helpers.
Replace their data-bearing `PlotOptions` calls with explicit `aggregate_means`
or `aggregate_rates`, then `prepare_geography` and `plot_geography`. Use M3's
`assign_parent` and native dissolution before rendering parent areas. Legacy
`normalize`, background/tiles, precision and weight-plot options are removed;
shared scales, distinct labels and explicit support selection replace them.
There are no compatibility shims. Preserve summary `attrs`: without metadata,
untagged columns are assumed to measure the same quantity; render unrelated
quantities in separate calls.

Verification on the existing CPython 3.12.7 full stack (Matplotlib 3.11.2,
GeoPandas 1.2.0, mapclassify 2.11.0, HoloViews 1.23.2, GeoViews 1.15.1 and
Cartopy 0.25.0):

- `.venv/bin/python -m pytest -q` — 66 pass. Checks inspect prepared numbers,
  actual native color mappers, class membership, support scales, polygon counts,
  opacity, missing regions, hover fields, contrast, layouts, projection and exports.
- `.venv/bin/ruff check src tests examples/prepared_comparison.py` and
  `.venv/bin/ruff format --check src tests examples/prepared_comparison.py` pass.
- A fresh-process check imports and renders statically without loading HoloViews,
  GeoViews, Bokeh, Panel, Cartopy, Seaborn or Contextily. Matplotlib rcParams stay
  unchanged, and interactive plots retain the existing renderer theme.
- `.venv/bin/python examples/prepared_comparison.py` exports continuous and
  classified PNG and standalone HTML comparisons to `/tmp/geomapviz-m4`.
  Both PNGs were visually inspected. The independent M1 arithmetic, coverage
  and native exports also pass in `/tmp/geomapviz-m4/native-baseline`.
- `git diff --check` passes. No dependencies were added; mapclassify reports
  its existing pure-Python fallback because optional Numba is absent.

M4 is complete. M5 remains the next implementation milestone. The separate ty
follow-up is still pending; historical resources and dependency declarations
remain M5 work, and the newer Cartopy compatibility/fresh-install verification
remains M6 work. Existing Ruff migration edits and the user-supplied PRD are
preserved separately. No push or publication.


## M5 — complete: optional interaction and lean distributions (2026-10-05)

The base package declares only NumPy, pandas, GeoPandas, Matplotlib and
mapclassify, supporting numerical preparation and continuous/classified static
maps. The `interactive` extra declares the renderer's direct imports:
HoloViews, GeoViews, Bokeh and Cartopy. Panel remains a transitive interactive
dependency; SciPy remains a transitive mapclassify dependency. Removed unused
direct declarations for Contextily, Panel, SciPy, Seaborn and tqdm. Existing
version bounds are retained; the supported-version matrix belongs to M6.

Interactive imports remain inside the requested rendering path. Missing direct
interactive dependencies raise an install instruction for
`geomapviz[interactive]`; unrelated missing modules preserve their original
exception. Both static and interactive rendering preserve Matplotlib settings,
and interactive rendering preserves an application-supplied Bokeh theme.

Removed all 16 historical boundary, raster and sample-data files from the
runtime source package: 148,975,966 bytes. No retained workflow uses them, and
no established provenance/example case justified moving them elsewhere. Removed
the package-data glob and disabled automatic wheel package-data inclusion.
The wheel contains only the five Python modules plus distribution metadata;
the source archive contains source, tests and normal build metadata, with no
historical runtime assets. The existing setuptools backend and dynamic version
source remain unchanged.

Breaking changes: removed `load_shp`, `load_geometry`, `merge_zip_df` and
`convert_category_to_code`, with no compatibility shims. Read caller-supplied
files with `geopandas.read_file`, then use the validated M3 preparation API.
The shared metric-list validator remains. The README gives base/extra checkout
installation commands and identifies the hosted documentation as 1.x.
`examples/prepared_comparison.py` now runs statically by default; add
`--interactive` to also export HTML. Documentation-site and CI work remain deferred.

Measured local CPython 3.12.7/Linux installations from wheels, with the same
resolver/index snapshot and no development extras:

| Runtime | Resolved dependencies, excluding Geomapviz | Installed dependency files (bytes) |
| --- | ---: | ---: |
| Before M5 | 61 | 791,717,169 |
| M5 base | 25 | 478,849,391 |
| M5 interactive | 52 | 667,392,654 |

Installed bytes sum unique files listed in dependency distribution metadata;
they exclude Geomapviz, generated bytecode, the interpreter and environment
bootstrap files. These are local file-size measurements, not download-size,
installation-speed or cross-platform claims.

| Artifact | Before M5 compressed bytes | M5 compressed bytes |
| --- | ---: | ---: |
| Wheel | 18,123,345 | 14,880 |
| Source distribution | 17,988,903 | 22,908 |

The original asset glob packaged 123 MB of the 149 MB source assets; it excluded
the historical `.grd` file. Both artifact pairs were built with `uv build`
using the same setuptools backend. Stale generated metadata was cleared before
the M5 build. The measured builds include the preserved working-tree Ruff edits.

Verification:

- `.venv/bin/python -m pytest -q` — 70 pass, including all earlier milestone
  cases, missing-extra errors for each direct interactive dependency, and a
  custom Bokeh theme checked before and after interactive construction.
- `.venv/bin/ruff check src tests examples/prepared_comparison.py`,
  `.venv/bin/ruff format --check src tests examples/prepared_comparison.py`, and
  `git diff --check` pass.
- Fresh wheel installations in `/tmp/geomapviz-m5/base` and
  `/tmp/geomapviz-m5/interactive` pass `uv pip check`. From a separate temporary
  working directory, hand-worked rate/support/coverage assertions and native
  continuous/classified PNG exports pass. The base installation has none of
  HoloViews, GeoViews, Cartopy, Bokeh or Panel installed or imported; requesting
  interaction gives the extra-install instruction. The interactive installation
  exports standalone continuous/classified HTML, preserving Matplotlib settings
  and the supplied Bokeh theme. Package paths point inside those environments.
- The comparison example runs in both fresh installations. Exports are in
  `/tmp/geomapviz-m5/base-example` and `interactive-example`. Verification scripts,
  dependency resolutions, baseline installation and artifacts remain under
  `/tmp/geomapviz-m5`; no checkout-relative runtime assets are used.
- Fresh interactive resolution uses Cartopy 0.26.0, GeoViews 1.15.1, HoloViews
  1.23.2 and Bokeh 3.9.2. The retained prepared EPSG:3857 renderer exports
  successfully on this combination. The older M1 hvPlot/PlateCarree failure is
  separate; M6 still owns broader compatibility and source-artifact installations.
- Wheel and source-archive inspections confirm that the runtime package contains
  only Python source, with no bundled boundaries, rasters or sample data.

M5 is complete. M6 remains the next implementation milestone; the separate ty
follow-up is also pending. Existing Ruff migration edits and the user-supplied
PRD are preserved separately. No push or publication.

## M6 — complete: local distribution and compatibility verification (2026-10-06)

Built a wheel and source archive with the existing setuptools backend and
installed both outside the repository, in fresh base and interactive environments
on CPython 3.12.7 and 3.14.8. The development version remains `2.0.0.dev0`;
this is local readiness evidence, not a published release candidate.

The [official Python downloads page](https://www.python.org/downloads/) identified
3.14.8 as the newest stable Python during verification. uv 0.12.15 had no binary
for that patch version, so the official CPython source was built in
`/tmp/geomapviz-m6`, with development headers extracted there. No system packages
or existing interpreters were replaced. Verification is for standard CPython
on Linux x86_64/glibc 2.39; other operating systems and free-threaded Python have
not been tested.

Set conservative dependency floors from a passing older stack rather than
retaining unverified 1.x bounds. These are supported tested floors, not a claim
that every earlier version fails. No upper bounds or new dependencies were added.

| Direct dependency | Tested floor on Python 3.12.7 | Current stack on Python 3.12.7 and 3.14.8 |
| --- | --- | --- |
| NumPy | 1.26.4 | 2.5.3 |
| pandas | 2.2.3 | 3.0.6 |
| GeoPandas | 1.0.1 | 1.2.0 |
| Matplotlib | 3.8.4 | 3.11.2 |
| mapclassify | 2.6.1 | 2.11.0 |
| HoloViews (extra) | 1.20.2 | 1.23.2 |
| GeoViews (extra) | 1.14.1 | 1.15.1 |
| Bokeh (extra) | 3.6.3 | 3.9.2 |
| Cartopy (extra) | 0.24.1 | 0.26.0 |

Floors were exercised together with resolved transitive dependencies; the
matrix does not claim every possible version combination. The current prepared
EPSG:3857 renderer passes on Cartopy 0.26.0 with GeoViews 1.15.1. M1's older
hvPlot/PlateCarree example failure does not require a core Cartopy pin. Context7
refreshed the native [GeoPandas active-geometry contract](https://github.com/geopandas/geopandas/blob/main/doc/source/docs/user_guide/data_structures.rst)
and [GeoViews/Cartopy installation guidance](https://github.com/holoviz/geoviews/blob/main/doc/index.rst).

Added `examples/verify_install.py`, a runnable check using the existing synthetic
sample and independent native arithmetic. It asserts the package comes from the
selected environment, numerical imports avoid rendering, weighted/rate/parent
arithmetic matches hand-worked values, coverage retains unobserved areas, caller
records are unchanged, and continuous/classified PNG export works. Interactive
mode also exports standalone HTML and preserves the application Bokeh theme and
Matplotlib settings. Base mode verifies optional distributions are absent and
interaction raises the install instruction. It prints actual dependency versions.

The source archive's plot tests imported examples that it previously omitted.
`MANIFEST.in` now includes the three synthetic Python examples, enabling the
included suite to run from the extracted archive. The wheel still contains only
five Python modules and distribution metadata; neither artifact contains country
boundaries, rasters or sample-data bundles. Archive inspection compared the wheel
modules with checkout source and checked the source examples/tests explicitly.
Final compressed sizes are 15,821 bytes for the wheel and 28,559 for the source
archive. Their README/metadata include the preserved working-tree Ruff edits.

| Installed artifact / stack | Python 3.12.7 | Python 3.14.8 |
| --- | --- | --- |
| Wheel, base | 39 tests and static installation check pass | 39 tests and static installation check pass |
| Source archive, base | 39 tests and static installation check pass | 39 tests and static installation check pass |
| Wheel, interactive | 70 tests and static/interactive installation check pass | 70 tests and static/interactive installation check pass |
| Source archive, interactive | 70 tests and static/interactive installation check pass | 70 tests and static/interactive installation check pass |
| Wheel, tested dependency floors with interaction | 70 tests and static/interactive installation check pass | Not tested |

`uv pip check` passes in every environment. Regression tests ran from the actual
extracted source archive, with
`PYTHONPATH` removed; package paths point into the selected environment. The base
subset excludes the geography test that also constructs interactive maps; that
case passes in every full-stack run. Every base environment retains no HoloViews,
GeoViews, Bokeh, Cartopy or Panel. Exports, resolved versions, logs, interpreter,
and artifacts remain in `/tmp/geomapviz-m6`.

Reproduction from a checkout, using the intended interpreter for each environment:

```sh
uv build --out-dir /tmp/geomapviz-artifacts
uv venv /tmp/geomapviz-check --python 3.12
uv pip install --python /tmp/geomapviz-check/bin/python /tmp/geomapviz-artifacts/geomapviz-2.0.0.dev0-py3-none-any.whl
mkdir -p /tmp/geomapviz-source
tar -xzf /tmp/geomapviz-artifacts/geomapviz-2.0.0.dev0.tar.gz -C /tmp/geomapviz-source
cd /tmp/geomapviz-source/geomapviz-2.0.0.dev0
/tmp/geomapviz-check/bin/python examples/verify_install.py --output /tmp/geomapviz-exports
```

For source installation, substitute the `.tar.gz` artifact. For interaction,
install `geomapviz[interactive] @ file:///tmp/geomapviz-artifacts/geomapviz-2.0.0.dev0-py3-none-any.whl`
and add `--interactive` to the check. Use a new environment for every variant.
After the installation check, install pytest and run `python -m pytest -q tests`
with the full stack. The base numerical/geography subset is:

```sh
python -m pytest -q tests/test_aggregation.py tests/test_geography.py -k 'not native_file_loading_and_interactive_renderer_transform_coordinates'
```

Ruff lint/format checks on `src tests examples` and `git diff --check` pass.
The README records the complete 2.0 migration: explicit means/rates and geography
preparation replace the old combined API, unsupported intervals and resource
loaders are removed, plotting options contain no data, and interaction is optional.
Strict cohort validation, undefined ratios and unmatched-boundary missing values
remain deliberate behavior. Preserve summary metadata for derived names and
quantity scales; CSV does not retain it. No compatibility shims were added.

The M1 maintenance decision still rests on the confirmed exposure/rate workflow
and reusable numerical/geographic validation. Native plotting remains sufficient
for rendering alone; synthetic evidence does not establish production adoption
or broad demand. No anonymized production case was supplied.

Remaining limitations: dependency/platform coverage is the matrix above; existing
mapclassify pure-Python fallback and older-stack upstream deprecation warnings
remain. Setuptools still warns about the legacy license table; modern SPDX
metadata can be addressed in the separate release-delivery phase. Hosted/site
documentation still describes 1.x. The separately recorded ty follow-up, site
work, CI/CD and release publication remain outside M6.

M0–M6 are complete. Existing Ruff migration changes and untracked PRD.md remain
separate. No push, tag, publication or external service change was performed.


## Documentation and CI phase — complete: Zensical 2.0 guide (2026-10-06)

Replaced the branch's Sphinx site with five hand-authored Markdown pages:
Quickstart, Aggregation, Geography, Plotting and export, and API and migration.
All pages and the site header identify this as **2.0 development documentation**.
The README now describes the current purpose, checkout installation, a runnable
example and documentation links. The original migration table is in the API guide.
Removed obsolete Sphinx configuration, reStructuredText, notebook, duplicated
requirements and legacy screenshots/logo. Released tags and the published 1.x
site were not changed; the existing hosted `/en/latest/` URL still returns 200.

The contributor `docs` dependency group replaces the old `doc` extra and pins
Zensical 0.0.68. There is no new runtime dependency, lockfile, custom theme,
notebook execution or API-generation plugin. Runtime Python modules and the
`interactive` extra are unchanged. The two committed example assets are copied
from the existing prepared workflow's continuous exports: `comparison.png`
and `comparison.html`. Trailing line whitespace in the generated HTML (bundled license comments and
blank lines) is trimmed before committing; the site build only copies the assets.

`zensical.toml` provides explicit navigation, the default theme and the canonical
`https://geomapviz.readthedocs.io/` URL. Strict validation rejects missing pages
and anchors. In this pinned release, non-Markdown asset targets are not checked
by the builder; their existence, HTTP responses, image loading and export
rendering were verified separately. See the official
[Zensical validation guide](https://zensical.org/docs/setup/validation/).

Read the Docs now uses Ubuntu 24.04/Python 3.12, bootstraps uv 0.12.15, installs
only the docs group, builds with `zensical build --clean`, and copies `site/.`
into `$READTHEDOCS_OUTPUT/html/`. The exact YAML-parsed install/build/copy commands
passed locally in a fresh simulated Read the Docs environment. Command scalars
containing `:all:` are quoted so YAML treats them as strings. This follows the
[Read the Docs custom-build guide](https://docs.readthedocs.com/platform/stable/intro/zensical.html);
no external project settings or default version were changed.

CI replaces the old Conda block with uv installation of the local editable
package and its `interactive`, `lint` and `test` extras. With uv 0.12.15,
`--only-binary=:all:` alone rejects the local build; the verified command adds
`--no-binary=geomapviz`, keeping that one build allowed and all dependencies
binary-only. Existing Ruff gates, `pytest --cov=./ --cov-report=xml` and Codecov
delivery are preserved.

### Wheel availability and platform limits

Refreshed wheel-only resolution using uv 0.12.15 and the actual base plus
interactive requirements from `pyproject.toml`. These are availability checks,
not runtime tests or a promise of all possible versions on those platforms.

| Target | Python 3.12 | Python 3.14 |
| --- | --- | --- |
| Linux x86_64, glibc 2.28+ | Pass | Pass |
| macOS Intel, macOS 13+ | Pass | Pass |
| macOS Apple Silicon | Pass on macOS 13 | Pass on macOS 14+ |
| Windows x86_64 | Pass | Pass |

Apple targets used `MACOSX_DEPLOYMENT_TARGET=13.0`, except Apple Silicon with
Python 3.14, which used `14.0`. The macOS 13 Apple Silicon/Python 3.12 resolution
selects GeoPandas 1.1.4 and pyproj 3.6.1. Intel macOS selects pyproj 3.7.2 on both
Python versions. Other cells select GeoPandas 1.2.0 and pyproj 3.8.0. The other
versions in the runtime snapshot below are shared by all eight resolution cells.
Windows also resolves colorama 0.4.6 and tzdata 2026.5, which are not needed on
the Linux installation. These platform additions are included in its resolution logs.

Actual fresh installations and rendering in this phase were on standard
CPython 3.12.7, Linux x86_64/glibc 2.39. M6's earlier Python 3.14 runtime evidence
remains separate. macOS, Windows, musl, ARM Linux and free-threaded Python were
not runtime-tested here. Wheel availability depends on Python, architecture,
OS minimums and future releases; no interpreter compilation was needed in this
phase. M6's source compilation was for its Python interpreter, not GeoPandas or
Cartopy. Prefer Python 3.12 with pip/uv; conda-forge remains an optional fallback
for platforms lacking wheels. No demonstrated need justified a Pixi environment.
[GeoPandas installation](https://geopandas.org/en/stable/getting_started/install.html),
[Cartopy installation](https://cartopy.readthedocs.io/stable/installing.html), and
[Pixi's uv comparison](https://pixi.prefix.dev/latest/switching_from/uv/)
provide the native-dependency context.

### Resolved versions

The Linux base has 25 dependency distributions; interaction has 52. Both use
Geomapviz `2.0.0.dev0`. Exact interactive runtime snapshot (base dependencies are
contained in this list):

```text
bokeh==3.9.2 cartopy==0.26.0 certifi==2026.7.22 charset-normalizer==3.5.2
cloudpickle==3.1.2 colorcet==3.2.1 contourpy==1.4.0 cycler==0.12.1 fonttools==4.66.1
geopandas==1.2.0 geoviews==1.15.1 holoviews==1.23.2 idna==3.20 jinja2==3.1.6
joblib==1.6.0 kiwisolver==1.5.1 linkify-it-py==2.2.0 mapclassify==2.11.0 markdown==3.11
markdown-it-py==4.2.0 markupsafe==3.0.4 matplotlib==3.11.2 mdit-py-plugins==0.6.1
mdurl==0.1.2 narwhals==2.26.0 networkx==3.7 nh3==0.3.7 numpy==2.5.3 packaging==26.3
pandas==3.0.6 panel==1.9.4 panel-material-ui==0.16.0 param==2.4.2 pillow==12.3.0
pyogrio==0.13.0 pyparsing==3.3.3 pyproj==3.8.0 pyshp==3.1.6 python-dateutil==2.9.0.post0
pyviz_comms==3.0.6 pyyaml==6.0.3 requests==2.34.2 scikit-learn==1.9.1 scipy==1.18.1
shapely==2.1.2 six==1.17.0 threadpoolctl==3.7.0 tornado==6.5.10 tqdm==4.70.1
typing_extensions==4.16.0 urllib3==2.8.0 xyzservices==2026.9.1
```

The docs-only builder environment contains the following 11 distributions and none of
Geomapviz, NumPy, pandas, GeoPandas, Matplotlib, mapclassify, HoloViews, GeoViews,
Cartopy, Bokeh or Panel:

```text
click==8.5.0 deepmerge==3.0.1 jinja2==3.1.6 markdown==3.11 markupsafe==3.0.4
pathspec==1.1.1 pygments==2.21.0 pymdown-extensions==12.1 pyyaml==6.0.3
tomli==2.4.1 zensical==0.0.68
```

The simulated Read the Docs environment additionally has bootstrap pip 26.2.1
and uv 0.12.15, with the same documentation dependencies and no runtime stack.

The CI check also resolved Ruff 0.16.10, pytest 9.1.1, pytest-cov 7.1.0 and
coverage 7.16.2. Browser verification used Playwright 1.63.0 with the existing
Chromium headless shell 1217 in a separate temporary environment; it is not a
project dependency.

### Verification and reproduction

All verification files, complete wheel-resolution logs, browser screenshots and
exports remain under `/tmp/geomapviz-docs` for this local session.

```sh
# Fresh docs-only environment; no runtime stack or project installation.
uv venv /tmp/geomapviz-docs/docs --python 3.12
uv pip install --python /tmp/geomapviz-docs/docs/bin/python --only-binary=:all: --group docs
/tmp/geomapviz-docs/docs/bin/zensical build --clean

# Fresh CI environment; dependencies must be wheels.
uv venv /tmp/geomapviz-docs/interactive --python 3.12
uv pip install --python /tmp/geomapviz-docs/interactive/bin/python --only-binary=:all: --no-binary=geomapviz -e '.[interactive,lint,test]'
/tmp/geomapviz-docs/interactive/bin/ruff check src tests examples
/tmp/geomapviz-docs/interactive/bin/ruff format --check src tests examples
/tmp/geomapviz-docs/interactive/bin/python -m pytest --cov=./ --cov-report=xml

# Generate documentation assets from the existing synthetic workflow.
/tmp/geomapviz-docs/interactive/bin/python examples/prepared_comparison.py --interactive --output /tmp/geomapviz-docs/example-exports

# Build and install artifacts into separate environments.
uv build --out-dir /tmp/geomapviz-docs/artifacts
uv venv /tmp/geomapviz-docs/base --python 3.12
uv pip install --python /tmp/geomapviz-docs/base/bin/python --only-binary=:all: /tmp/geomapviz-docs/artifacts/geomapviz-2.0.0.dev0-py3-none-any.whl
uv venv /tmp/geomapviz-docs/wheel-interactive --python 3.12
uv pip install --python /tmp/geomapviz-docs/wheel-interactive/bin/python --only-binary=:all: 'geomapviz[interactive] @ file:///tmp/geomapviz-docs/artifacts/geomapviz-2.0.0.dev0-py3-none-any.whl'

# Repeat with each platform/Python cell; set MACOSX_DEPLOYMENT_TARGET on macOS.
uv pip install --python /tmp/geomapviz-docs/docs/bin/python --dry-run --only-binary=:all: --python-version 3.12 --python-platform x86_64-manylinux_2_28 -r pyproject.toml --extra interactive
```

- Docs-only build reports no issues. Disposable copies containing a missing
  Markdown page or anchor exit with status 1 in strict mode.
- Headless Chromium checks all five pages at 1440×1000 and 390×844: navigation,
  local links/resources, same-page anchors, tables with headers, image alt text
  and successful image loading. Pages have no viewport overflow; code blocks
  scroll inside their containers. Screenshots were inspected. Both layouts open
  the HTML example without JavaScript errors. With all HTTP(S) requests blocked,
  the file export still renders from its inline resources. The HTML example is
  a fixed three-column layout, so narrow screens scroll horizontally; the guide
  explains `ncols=1` for a narrower export.
- All 12 runnable Python snippets from Quickstart, Aggregation, Geography,
  Plotting/export and README execute against the fresh installed interactive
  wheel. The caller-file example uses a temporary GeoPackage made from the
  same synthetic boundaries. Arithmetic assertions and PNG/HTML exports pass.
- Fresh base and interactive wheel installations pass `uv pip check` and the
  existing `examples/verify_install.py` from copied examples outside the checkout
  with `PYTHONPATH` removed. Package paths point into their environments. Base
  mode has no interactive distributions; both modes export continuous/classified
  PNGs, and interactive mode also exports inline HTML. The same checks retain
  numerical import independence, parent arithmetic and application plotting styles.
- Full interactive coverage command: **70 passed**, with the 12 existing
  mapclassify pure-Python fallback warnings. The preserved whole-checkout coverage
  scope records 888/957 lines (92.79%); it includes tests/examples, not only runtime
  source. Ruff lint/format checks pass on `src tests examples`.
- Final wheel: 14,683 bytes; source archive: 25,939 bytes. The wheel has exactly
  five runtime modules plus metadata. Module bytes match checkout source; README
  metadata matches the final guide links. The source archive retains examples and
  tests; neither artifact bundles documentation exports or legacy geographic assets.
  The obsolete `doc` extra is absent. The existing setuptools license-table
  deprecation warning remains outside this migration.
- `git diff --check` passes. No runtime implementation or test files were changed.

This completes one local documentation/CI migration phase on `feat/v2.0.0`.
Pushing, merging, activating a Read the Docs development preview, changing its
published default version and publishing 2.0 remain separate delivery actions.
When authorized, preview the feature branch and verify hosted navigation and
asset downloads before changing the default version. The separate ty follow-up
remains pending.

## Release plan, Phase 1 — local CI and ty implementation complete (2026-10-06)

Updated `.github/workflows/ruff.yml` to test Linux Python 3.12 and 3.14. Both
jobs install the `interactive`, `lint` and `test` extras, check Ruff lint/format
on `src tests examples`, check types on `src`, and run the existing test suite.
The uv 0.12.15 installer and wheel-only policy are retained, including the local
editable Geomapviz build exception. Added `ty==0.0.84` to `lint` and
`[tool.ty.src] include = ["src"]`; the workflow explicitly resolves imports
against its `.venv`. No runtime source, public API or regression tests changed.

Only Python 3.12 generates coverage, now with `--cov=geomapviz`. A named
`coverage` artifact transfers `coverage.xml` to a separate Codecov job after
both Python jobs pass. Only that job has `id-token: write`; it uses OIDC,
`files: coverage.xml`, `disable_search: true` and `fail_ci_if_error: true`.
The pinned action natively disables OIDC for fork PRs, selects their tokenless
branch label, and associates PR uploads with the head commit. No token secret
or custom branch override is configured. Dependabot actors and PR authors skip
artifact/Codecov uploads while retaining lint, type and test checks.

Both jobs have 20-minute timeouts; superseded runs on the same workflow/ref are
cancelled, and both checkouts disable persisted credentials. Added weekly
GitHub Actions updates to the existing Dependabot configuration.

Release tags were resolved through the GitHub API to immutable commits:

- checkout 7.0.1: `3d3c42e5aac5ba805825da76410c181273ba90b1`.
- setup-python 7.0.0: `5fda3b95a4ea91299a34e894583c3862153e4b97`.
- Codecov 7.1.1: `303a32d7a59b442fa8d48b6a1cc6825c09c847a5`
  (peeled from its annotated release tag).
- upload-artifact 7.0.1: `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a`.
- download-artifact 8.0.1: `3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c`.

Verification used fresh temporary environments under
`/tmp/geomapviz-release-phase1`, with uv 0.12.15 and the same wheel-only editable
install command as CI. CPython 3.12.7 and 3.14.8 each installed 61 distributions;
`uv pip check` passes on both. Ruff 0.16.10 lint and format checks pass on all
11 source/test/example files, and ty 0.0.84 passes against both environments.
No type suppressions or source repairs were necessary.

- Python 3.12.7: `python -m pytest --cov=geomapviz --cov-report=xml` —
  **70 passed**, 12 existing mapclassify pure-Python fallback warnings.
  The XML records **364/379 lines (96.04%)**, containing only the five runtime
  modules under `src/geomapviz`; the local report is at
  `/tmp/geomapviz-release-phase1/coverage.xml`.
- Python 3.14.8: `python -m pytest` — **70 passed**, the same 12 warnings.
- `actionlint .github/workflows/ruff.yml` passes with actionlint 1.7.12;
  `git diff --check` passes.
- The pinned Codecov action's actual fork/branch/commit shell steps passed
  synthetic push, same-repository PR and fork PR checks. Its OIDC guard was
  inspected to confirm forks do not request an identity token. This verifies
  local routing, not authentication or acceptance by Codecov.

Configuration references:
[ty environment discovery](https://docs.astral.sh/ty/type-checking/#environment-discovery),
[Codecov authentication](https://docs.codecov.com/docs/codecov-tokens), and
[the pinned Codecov action](https://github.com/codecov/codecov-action/blob/v7.1.1/action.yml).

Phase 1 implementation and local checks are complete. Live GitHub matrix runs,
artifact transfer and successful Codecov delivery for the correct commit still
require a pushed run; none is claimed here. Stop after this phase. Phase 2
metadata/publishing setup, the hosted preview and the 2.0 release remain pending.
