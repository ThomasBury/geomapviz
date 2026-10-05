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
