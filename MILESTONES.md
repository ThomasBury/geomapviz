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
