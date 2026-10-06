# Geomapviz 2.0: geographic model comparison

Status: proposed; implementation has not started.

Current release: 1.1.3.

Target release: 2.0.0, subject to the relevance decision in milestone M1.

## 1. Purpose and product decision

Geomapviz should help model analysts compare observations and predictions by
geographic area, understand how much evidence supports each comparison, and
inspect the numbers behind a map. The package should earn its maintenance cost
through a useful recurring workflow.

Generic choropleth plotting is already available in GeoPandas, hvPlot/GeoViews,
and Plotly. A standalone package is justified only if it makes geographic model
comparison materially easier and safer than composing those tools directly.

The primary audience is model analysts, including actuarial analysts working with
exposure-weighted observations and predictions. The maintenance criterion is a
distinct useful workflow. Popularity and the age of the latest release are
supporting evidence, rather than sufficient reasons to continue or stop.

Before substantial development, compare one representative analyst task with a
small implementation using existing libraries. Record the repeated work,
correctness risks, user benefit, and remaining maintenance cost. If no concrete
workflow justifies the package, stop the 2.0 work and prepare a deprecation and
archive recommendation.

## 2. Scope

Included:

- Trustworthy aggregation of observations and predictions by geographic ID.
- Explicit distinctions between weighted means, totals, and rates.
- Validated joins to user-supplied boundaries and aggregation to larger areas.
- Comparable maps, support information, and inspectable numerical results.
- A small public API, optional interactive plotting, package dependency cleanup,
  and local verification of built distributions.

Deferred:

- Documentation-site work and CI/CD configuration.
- New confidence-interval methods until a concrete use case and statistical
  assumptions are established.
- Spatial regression, spatial dependence tests, model fitting, dashboards,
  automatic boundary downloads, and additional plotting backends.
- Publishing releases, changing external services, or archiving the repository.
  These are separate delivery actions after the package decision.

## 3. Current problems to address

The package review identified the following problems:

- Missing observations retain their weights in the averaging denominator;
  entirely missing groups can become zero.
- Negative weights are accepted, and internal names collide with user columns
  such as `count` and `weight`.
- Categorical geographic IDs become integer codes, leading zeros can be lost,
  and the geometry join normalizes the wrong summary table.
- Confidence intervals use unsupported variability assumptions and clip bounds
  using unrelated groups. They are calculated even when uncertainty is not
  requested for the plot.
- Interactive polygons are labelled as Web Mercator regardless of their actual
  coordinate reference system (CRS).
- The interactive constructor option and one-column facet layouts fail;
  normalized static plots draw layers twice, and white backgrounds receive
  white titles.
- Tests reference a deleted module and the previous API.
- Static plotting imports and initializes the interactive stack and changes
  global plotting styles. Resource loading relies on `pkg_resources`, which
  modern setuptools has removed.
- Historical country boundaries and large raster backgrounds create avoidable
  maintenance and installation costs. Italy is advertised without a bundled file.

Numerical and option failures were reproduced with selected unchanged source
definitions using real pandas, NumPy, and Matplotlib. Full geospatial rendering
and a fresh installation remain unverified because dependencies were unavailable
and the attempted PyPI download timed out. Reproduce those paths in a complete
environment before treating a repair as verified.

## 4. Required behavior

### Aggregation and comparison

- Accept explicit geographic IDs, metric column names, and an optional weight
  column name. Preserve metric names and avoid internal column collisions.
- Preserve categorical ID labels and leading zeros. Reject missing IDs; never
  replace identifiers with category codes.
- Weighted means use `sum(weight * value) / sum(weight)` on a defined cohort.
  Without a weight column, use equal weights.
- Reject missing or non-finite observed/predicted values by default. Callers must
  select a valid cohort explicitly so models are compared on the same records.
- Require finite, non-negative weights. Zero-weight rows contribute neither to
  averages nor support counts. Reject groups with no positive total weight and
  identify the affected areas in the error.
- Return ordinary pandas summaries independently of plotting. Include support
  counts and total weight, with names that cannot overwrite user metrics.
- For a rate workflow confirmed in M1, distinguish observed totals from existing
  rates. Observed rates are `sum(observed_total) / sum(exposure)`; predictions
  supplied as rates are exposure-weighted. Exposure must not be applied twice.
- For that confirmed workflow, provide signed observed-minus-predicted
  differences and observed-to-expected ratios from compatible aggregate values.
  An expected value of zero produces an undefined ratio, rather than infinity
  or a fabricated zero.
- Do not compute statistical intervals by default. Remove unsupported interval
  calculations from the new API; any future uncertainty feature requires its own
  methodological specification and validation.

### Geometry and coverage

- Accept a user-supplied GeoDataFrame as the primary boundary input. Preserve its
  CRS; transform coordinates when a renderer requires another projection.
- Validate boundary identifiers, required columns, CRS, and non-empty valid
  geometries. Report problems without silently guessing a CRS or repairing data.
- Normalize both join keys consistently without destroying textual identifiers.
  Require one boundary per base geographic ID, or explicit dissolution first.
- Reject observation IDs with no matching boundary, listing the affected IDs.
  Retain boundaries with no observations and represent their metrics as missing.
- Expose matched and unmatched boundary coverage in ordinary tabular or dictionary
  results so users can inspect it before plotting.
- Aggregation to a larger area must use an unambiguous mapping from each base ID
  to its parent area. Recompute metrics from underlying totals and weights;
  never take an unweighted average of area averages.
- Remove historical country bundles and raster imagery from the core distribution.
  Retain assets outside the runtime package only when their source, vintage, CRS,
  redistribution terms, and concrete example use are established.

### Plotting and public interfaces

- Rendering consumes prepared geographic summaries; it must also be possible to
  use those summaries directly with GeoPandas, hvPlot, or Plotly.
- Keep data preparation separate from rendering options. Align annotations and
  accepted inputs, including lists of metric names, weight column names, and
  the interactive flag.
- Return native Matplotlib figures for static output and native HoloViews/GeoViews
  objects for the retained interactive output. Reuse their export facilities.
- Maps comparing the same quantity share color limits or classification bins.
  Support counts and exposure use separate scales. Missing values remain visibly
  distinct from zero. Difference maps use a scale centered on zero.
- Draw each layer once. Respect requested opacity, use readable titles and legends,
  and support one-column and partially filled layouts.
- Handle empty input and constant-valued data clearly. Classification must not
  crash when there are fewer distinct finite values than requested bins.
- Importing the package or requesting a static plot must not initialize an
  interactive renderer or change application-wide plotting styles.

## 5. Engineering and compatibility

- Use small functions in the existing aggregation, geometry, and plotting modules.
  Reuse pandas group operations, GeoPandas joins/dissolve/projections, mapclassify,
  and the existing rendering libraries. Add dependencies only when a confirmed
  requirement cannot be met clearly with the current stack or standard library.
- Keep aggregation usable without plotting imports. Load the interactive stack
  only when requested and move it to an `interactive` optional dependency group.
- Remove unused direct dependency declarations. Verify the resolved dependency
  set before claiming fewer installed dependencies or faster installation.
- Remove `pkg_resources` and obsolete resource helpers. Use `importlib.resources`
  only for packaged resources that remain necessary. Keep the existing setuptools
  build backend and dynamic version source.
- Treat 2.0 as a breaking release: obsolete helpers, interval outputs, bundled-data
  loaders, and inconsistent options may be removed. Do not add compatibility
  shims without an identified consumer and a concrete migration need.
- Proposed Python minimum: 3.12. Test that minimum and the newest stable Python
  available during implementation. Declare dependency bounds based on tested
  compatibility, rather than speculative upper limits.
- Preserve caller data. Validate trust boundaries and use clear built-in errors
  that name the invalid column, area, or value.
- Use the project's pytest setup and existing formatting/lint tools. Do not make
  a tooling migration a prerequisite for repairing the package.
- Deliver each milestone as a focused, independently reviewable change with
  Conventional Commit messages and passing relevant local checks. Record any
  deliberate shortcut with a `ponytail:` comment naming its limit and replacement
  condition. Complete one milestone before starting the next.

## 6. Milestones

All milestones below are pending. Creating this PRD does not execute them.

### M0 — Create the next-version branch

**First action:** create `feat/v2.0.0` from the reviewed `main` checkout before
changing package code. Inspect the working tree and preserve unrelated changes.

```sh
git switch -c feat/v2.0.0
```

Record the base commit, version 1.1.3, environment, and baseline check results.
Keep the 1.x history available. Do not push or publish as part of this milestone.

**Done when:** the intended branch is active, its base is recorded, and baseline
failures are distinguished from failures introduced by subsequent work.

### M1 — Decide whether the package should continue

Compare one recurring geographic model-analysis task with native-library
implementations. Start with deterministic synthetic observations and polygons;
use a representative anonymized case when available. Synthetic examples establish
feasibility, but do not establish user demand.

Use GeoPandas/pandas as the static baseline and hvPlot/GeoViews as the interactive
baseline. Consider Plotly when it matches the intended user's existing workflow.
Evaluate statistical correctness, coverage checks, comparison quality, repeated
caller effort, installation cost, and package maintenance.

**Done when:** a maintainer decision records the concrete user benefit and the
minimum retained workflow. If the benefit is only a thin plotting wrapper, stop
M2–M6 and prepare a deprecation/migration recommendation. Archive execution and
its documentation are later work.

### M2 — Establish a trustworthy numerical contract

Set the development version to `2.0.0.dev0`. Repair shared aggregation and input
validation, preserve IDs and metric names, eliminate unsupported interval
calculations, and implement only the comparison metrics retained by M1.
Replace obsolete numerical tests with deterministic assertions against hand-worked
values.

**Done when:** weighted and unweighted results are correct; missing values,
invalid weights, zero-support groups, and name collisions behave as specified;
caller data remains unchanged; aggregation works independently of rendering.

### M3 — Make geography and coverage reliable

Repair key normalization and joins, validate boundary uniqueness and CRS, expose
coverage, and support correct aggregation to parent areas. Use tiny synthetic
polygons for checks instead of historical country-file shapes.

**Done when:** categorical IDs, leading-zero IDs, numeric IDs, unmatched areas,
duplicate boundaries, missing CRS, and parent-area aggregation have meaningful
checks; projection changes preserve geographic alignment.

### M4 — Make comparisons usable and consistent

Route static and interactive plots through the same prepared summaries. Repair
options, layouts, contrast, opacity, classification, and shared scales. Surface
support and missing-data information alongside the retained comparison metrics.

**Done when:** single maps and comparisons render on supported backends; one-column
and incomplete grids work; layers are drawn once; comparable panels use the same
scale; support maps use their own scale; native export works.

### M5 — Reduce package coupling and bundled assets

Make interactive dependencies optional, remove unnecessary declarations and import
side effects, remove obsolete resource loading and runtime assets, and remove
helpers. Keep only files required by the retained workflow.

**Done when:** a base installation supports the static workflow without the
interactive stack; an installation with the interactive extra supports interactive
maps; importing either path leaves global plotting settings unchanged; built
distributions contain only intended assets. Record dependency and artifact sizes.

### M6 — Verify the release candidate locally

Build a wheel and source distribution with the existing backend. Install each in
fresh environments outside the repository, run the retained end-to-end workflow,
and verify the supported Python/dependency combinations. Record breaking changes,
remaining limitations, and the evidence for the M1 relevance decision.

**Done when:** numerical, geometry, plotting, import, and distribution checks pass;
both installed artifacts work without checkout-relative files; the package is
ready for a separate documentation, CI/CD, and release-delivery phase.

## 7. Acceptance evidence

Maintain a small pytest suite covering real failure modes:

- A weighted mean of `[10, 20]` with weights `[1, 3]` equals `17.5`.
- Missing observations fail explicitly; negative/non-finite weights and
  zero-support groups cannot silently generate misleading values.
- Rate summaries match hand-computed totals divided by exposure, including
  unequal exposure and undefined observed-to-expected ratios where applicable.
- IDs retain categorical labels and leading zeros; joins preserve coverage and
  reject duplicate or unknown observation IDs.
- Aggregation to a parent area matches aggregation from the original records.
- The same polygons align before and after a required CRS transformation.
- Shared-scale plots preserve comparable values; missing regions differ from
  zero; constant data, few classes, and one-column layouts are handled.
- Core operations preserve input data, static imports avoid optional dependencies,
  and plotting does not change global styles.
- Built artifacts install and export a representative static and interactive
  comparison using only installed resources.

Milestone completion requires passing evidence for its own changes and the checks
from earlier milestones. Screenshots and return-type checks alone are insufficient.

## 8. Existing solutions and source references

- [GeoPandas aggregation](https://geopandas.org/en/stable/docs/user_guide/aggregation_with_dissolve.html)
  and [interactive mapping](https://geopandas.org/en/stable/docs/user_guide/interactive_mapping.html):
  primary native baselines for joins, aggregation, and maps.
- [hvPlot geographic data](https://hvplot.holoviz.org/en/docs/latest/user_guide/Geographic_Data.html),
  [layouts](https://hvplot.holoviz.org/en/docs/latest/user_guide/Subplots.html), and
  [color scales](https://hvplot.holoviz.org/en/docs/latest/ref/plotting_options/color_colormap.html):
  baseline for interactive comparisons.
- [Plotly choropleths](https://plotly.com/python/tile-county-choropleth/):
  an alternative for users already working with Plotly.
- [PySAL residual diagnostics](https://pysal.org/spreg/generated/spreg.MoranRes.html):
  an existing home for spatial statistical analysis outside this package's scope.
- [geoplot](https://github.com/ResidentMario/geoplot) is in maintenance mode;
  [splot](https://github.com/pysal/splot) is being archived as functionality moves
  into associated PySAL projects. Neither is a reason to build another general
  plotting layer.
- [Statbel geographic classifications](https://statbel.fgov.be/fr/propos-de-statbel/methodologie/classifications/geographie)
  and [setuptools history](https://setuptools.pypa.io/en/latest/history.html):
  evidence for boundary-vintage and resource-loader issues found in the review.

Refresh external capability and compatibility claims during M1 and M6 before
making a release or archive decision.
